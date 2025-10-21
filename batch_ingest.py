#!/usr/bin/env python3
import os
import io
import json
import re
import hashlib
import time
import random
import argparse
import logging
from typing import List, Dict, Tuple, Set, Optional, Any
from datetime import datetime
from difflib import get_close_matches
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.exceptions import ClientError
from botocore.config import Config as BotoConfig
from pypdf import PdfReader
from docx import Document as DocxDocument
import pandas as pd
import tiktoken
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec  # pylint: disable=no-name-in-module
from openai import OpenAI
from openai import APIError, RateLimitError

# ---------------- Env & constants ----------------
load_dotenv()

# ---- logging ----
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

# -------------- Config (edit or override via env/CLI) --------------
DEFAULT_BUCKET = os.getenv("BUCKET", "desopsus")
DEFAULT_PREFIX = os.getenv("PREFIX", "")
INDEX_NAME = os.getenv("INDEX_NAME", "operational-docs")
NAMESPACE = os.getenv("NAMESPACE") or None
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
DIMENSION = int(os.getenv("DIMENSION", "1536"))
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))
UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", "64"))
MAX_FILE_MB = float(os.getenv("MAX_FILE_MB", "0"))
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "30"))
SKIP_EXISTING = os.getenv("SKIP_EXISTING", "true").lower() == "true"
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))
MAX_PENDING_VECTORS = int(os.getenv("MAX_PENDING_VECTORS", "10000"))

EXT_WHITELIST = {"txt", "md", "csv", "json", "pdf", "docx"}

# Global cache for building names
BUILDING_NAME_CACHE = {}  # Maps normalized names to canonical names
BUILDING_ALIASES_CACHE = {}  # Maps aliases to canonical names

# Statistics tracking
stats = {
    "files_processed": 0,
    "files_skipped": 0,
    "files_failed": 0,
    "total_vectors": 0,
    "vectors_skipped": 0,
    "failed_files": [],
}

# ---------------- Clients ----------------
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    config=BotoConfig(
        connect_timeout=5,
        read_timeout=60,
        retries={"max_attempts": 5},
        max_pool_connections=50,
    ),
)
pc = Pinecone(api_key=PINECONE_API_KEY)
oai = OpenAI(api_key=OPENAI_API_KEY)
enc = tiktoken.get_encoding("cl100k_base")

# Ensure index exists and validate
existing = {i["name"] for i in pc.list_indexes()}
if INDEX_NAME not in existing:
    logging.info("Creating Pinecone index '%s' (dim=%d)...",
                 INDEX_NAME, DIMENSION)
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    # Validate dimension matches
    index_info = pc.describe_index(INDEX_NAME)
    if index_info.dimension != DIMENSION:
        raise RuntimeError(
            f"Index dimension mismatch: {index_info.dimension} != {DIMENSION}"
        )
    logging.info("Using existing index '%s' (dim=%d)", INDEX_NAME, DIMENSION)

index = pc.Index(INDEX_NAME)

# ---------------- Helpers ----------------


def backoff_sleep(attempt: int, base: float = 0.5, cap: float = 10.0):
    """Exponential backoff with jitter."""
    t = min(cap, base * (2**attempt)) * (0.5 + random.random())
    time.sleep(t)


def get_existing_ids(bucket: str) -> Set[str]:
    """
    Fetch all existing vector IDs for this bucket to avoid re-embedding.
    Uses pagination to handle large indexes.
    """
    if not SKIP_EXISTING:
        return set()

    logging.info("Fetching existing vector IDs to skip duplicates...")
    existing_set = set()
    try:
        # List all vector IDs (this may need pagination depending on index size)
        results = index.list(namespace=NAMESPACE)
        for id_list in results:
            existing_set.update(id_list)

        logging.info(
            "Found %d existing vectors to potentially skip", len(existing_set))
    except Exception as e:  # pylint: disable=broad-except
        logging.warning(
            "Could not fetch existing IDs, will process all: %s", e)
        return set()

    return existing_set


def list_s3_objects(bucket: str, prefix: str = "") -> List[Dict]:
    """List all S3 objects in a bucket with the given prefix, filtered by extension."""
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    out = []
    for p in pages:
        for obj in p.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            if ext(key) not in EXT_WHITELIST:
                continue
            out.append(obj)
    return out


def fetch_bytes(bucket: str, key: str) -> bytes:
    """Fetch object bytes from S3."""
    r = s3.get_object(Bucket=bucket, Key=key)
    return r["Body"].read()


def ext(key: str) -> str:
    """Get file extension from S3 key."""
    return key.lower().rsplit(".", 1)[-1] if "." in key else ""


def estimate_tokens(text: str) -> int:
    """Fast token estimation: ~4 chars per token."""
    return len(text) // 4


def load_building_names_with_aliases(bucket: str, key: str) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """
    Load building names and create comprehensive alias mappings.

    Returns:
        (canonical_names, name_to_canonical_map, alias_to_canonical_map)
    """
    global BUILDING_NAME_CACHE, BUILDING_ALIASES_CACHE

    if BUILDING_NAME_CACHE and BUILDING_ALIASES_CACHE:
        return list(set(BUILDING_NAME_CACHE.values())), BUILDING_NAME_CACHE, BUILDING_ALIASES_CACHE

    try:
        data = fetch_bytes(bucket, key)
        df = pd.read_csv(io.BytesIO(data))

        canonical_names = []
        name_to_canonical = {}
        alias_to_canonical = {}

        for _, row in df.iterrows():
            # Get primary property name
            primary_name = row.get("Property name")
            if pd.isna(primary_name):
                continue

            primary_name = str(primary_name).strip()
            canonical_names.append(primary_name)

            # Map normalized primary name to canonical
            normalized_primary = primary_name.lower().strip()
            name_to_canonical[normalized_primary] = primary_name

            # Process Property names (semicolon-separated)
            property_names = row.get("Property names")
            if pd.notna(property_names):
                for name in str(property_names).split(";"):
                    name = name.strip()
                    if name:
                        normalized = name.lower().strip()
                        alias_to_canonical[normalized] = primary_name
                        name_to_canonical[normalized] = primary_name

            # Process Property alternative names (semicolon-separated)
            alt_names = row.get("Property alternative names")
            if pd.notna(alt_names):
                for name in str(alt_names).split(";"):
                    name = name.strip()
                    if name:
                        normalized = name.lower().strip()
                        alias_to_canonical[normalized] = primary_name

            # Process UsrFRACondensedPropertyName (common abbreviations)
            condensed = row.get("UsrFRACondensedPropertyName")
            if pd.notna(condensed):
                condensed = str(condensed).strip()
                if condensed:
                    normalized = condensed.lower().strip()
                    alias_to_canonical[normalized] = primary_name

        # Update global caches
        BUILDING_NAME_CACHE = name_to_canonical
        BUILDING_ALIASES_CACHE = alias_to_canonical

        logging.info(
            "Loaded %d canonical building names with %d aliases",
            len(canonical_names),
            len(alias_to_canonical)
        )

        return canonical_names, name_to_canonical, alias_to_canonical

    except Exception as e:
        logging.warning("Could not load building names with aliases: %s", e)
        return [], {}, {}


def find_closest_building_name(
    extracted_name: str, known_buildings: List[str]
) -> str:
    """
    Find the closest matching building name using fuzzy matching.
    Uses multiple strategies: exact match, substring match, and fuzzy match.
    """
    if not extracted_name or not known_buildings:
        return extracted_name

    # Strategy 1: Try exact match first (case-insensitive)
    for building in known_buildings:
        if building.lower() == extracted_name.lower():
            return building

    # Strategy 2: Try substring match (extracted name contained in building name)
    for building in known_buildings:
        if extracted_name.lower() in building.lower():
            return building

    # Strategy 3: Try reverse substring match (building name contained in extracted name)
    for building in known_buildings:
        if building.lower() in extracted_name.lower():
            return building

    # Strategy 4: Use difflib for fuzzy match (80% similarity threshold)
    matches = get_close_matches(
        extracted_name, known_buildings, n=1, cutoff=0.8)
    if matches:
        return matches[0]

    # Return original if no good match found
    return extracted_name


def find_closest_building_name_enhanced(
    extracted_name: str,
    name_to_canonical: Dict[str, str],
    alias_to_canonical: Dict[str, str],
    known_buildings: List[str]
) -> str:
    """
    Enhanced building name matching using canonical names and aliases.

    Args:
        extracted_name: Building name extracted from query/filename
        name_to_canonical: Map of normalized names to canonical names
        alias_to_canonical: Map of aliases to canonical names
        known_buildings: List of canonical building names (for fallback)

    Returns:
        Matched canonical building name
    """
    if not extracted_name:
        return extracted_name

    normalized = extracted_name.lower().strip()

    # Strategy 1: Direct match in canonical names
    if normalized in name_to_canonical:
        canonical = name_to_canonical[normalized]
        logging.info("Exact match: '%s' -> '%s'", extracted_name, canonical)
        return canonical

    # Strategy 2: Match in aliases
    if normalized in alias_to_canonical:
        canonical = alias_to_canonical[normalized]
        logging.info("Alias match: '%s' -> '%s'", extracted_name, canonical)
        return canonical

    # Strategy 3: Substring match in canonical names
    for norm_name, canonical in name_to_canonical.items():
        if normalized in norm_name or norm_name in normalized:
            logging.info("Substring match: '%s' -> '%s'",
                         extracted_name, canonical)
            return canonical

    # Strategy 4: Substring match in aliases
    for alias, canonical in alias_to_canonical.items():
        if normalized in alias or alias in normalized:
            logging.info("Alias substring match: '%s' -> '%s'",
                         extracted_name, canonical)
            return canonical

    # Strategy 5: Fuzzy match (fallback to original function)
    result = find_closest_building_name(extracted_name, known_buildings)
    if result != extracted_name:
        logging.info("Fuzzy match: '%s' -> '%s'", extracted_name, result)

    return result


def extract_building_name_from_filename(
    filename: str, known_buildings: Optional[List[str]] = None
) -> str:
    """
    Extract building name from PDF/DOCX filename with improved accuracy.
    Handles multiple FRA naming patterns and BMS documents.
    """
    # Remove path and extension
    name = os.path.basename(filename)
    name = name.replace(".pdf", "").replace(".docx", "").replace(".doc", "")

    # Handle Fire Risk Assessment naming patterns
    # Pattern 1: FM-FRA-BuildingName-YYYY-MM
    fra_match = re.match(r"(?:FM-)?FRA-(.+?)-\d{4}-\d{2}", name, re.IGNORECASE)
    if fra_match:
        building_part = fra_match.group(1)
        # Convert camelCase or PascalCase to spaced format
        building_part = re.sub(r"([a-z])([A-Z])", r"\1 \2", building_part)
        building_part = re.sub(r"(\d)([A-Z])", r"\1 \2", building_part)
        name = building_part

    # Pattern 2: SRL-FRA-BuildingName-YYYY-MM (older checksheet style)
    srl_match = re.match(r"SRL-FRA-(.+?)-\d{4}-\d{2}", name, re.IGNORECASE)
    if srl_match:
        building_part = srl_match.group(1)
        building_part = re.sub(r"([a-z])([A-Z])", r"\1 \2", building_part)
        building_part = re.sub(r"(\d)([A-Z])", r"\1 \2", building_part)
        name = building_part

    # Handle BMS documents
    # Pattern: UoB-BuildingName-BMS-...
    bms_match = re.match(r"UoB-(.+?)-BMS", name, re.IGNORECASE)
    if bms_match:
        building_part = bms_match.group(1)
        name = building_part

    # Remove UoB prefix for standard documents
    name = name.replace("UoB-", "")

    # Generic patterns: remove common technical terms and suffixes
    patterns_to_remove = [
        r"-BMS.*$",
        r"-Controls.*$",
        r"-O&M.*$",
        r"-OM.*$",
        r"-Des-Ops.*$",
        r"-DesOps.*$",
        r"-Prototype.*$",
        r"-Manual.*$",
        r"-Rev\d+.*$",
        r"-rev\d+.*$",
        r"-P\d+.*$",
        r"-As-Installed.*$",
        r"-Project.*$",
        r"-iiq.*$",
        r"-trend.*$",
        r"-AC-to.*$",
        r"-FRA.*$",
        r"-SRL.*$",
        r"-\d{4}-\d{2}$",
        r"-v\d+$",
        r"-draft$",
    ]

    for pattern in patterns_to_remove:
        name = re.sub(pattern, "", name, flags=re.IGNORECASE)

    # Replace hyphens with spaces
    name = name.replace("-", " ")

    # Clean up multiple spaces and trim
    name = re.sub(r"\s+", " ", name).strip()

    # Capitalize properly
    name = " ".join(word.capitalize() for word in name.split())

    return name


def is_fire_risk_assessment(key: str, text: str = "") -> bool:
    """
    Determine if a document is a Fire Risk Assessment.
    Checks filename and content for multiple patterns.
    """
    # Check filename patterns
    key_lower = key.lower()

    # Common FRA filename patterns
    fra_patterns = [
        "fra",
        "fire-risk",
        "fire_risk",
        "srl-fra",
        "fm-fra",
    ]

    if any(pattern in key_lower for pattern in fra_patterns):
        return True

    # Check content for FRA indicators (first and last sections)
    if text:
        text_sample = text[:3000] + (text[-1000:] if len(text) > 4000 else "")
        fra_indicators = [
            "fire risk assessment",
            "regulatory reform (fire safety) order",
            "fire safety order",
            "date of fire risk assessment",
            "risk level indicator",
            "fire checksheet",
            "bristol university fire checksheet",
            "date of inspection",
            "recommended reinspection date",
        ]
        text_lower = text_sample.lower()
        if any(indicator in text_lower for indicator in fra_indicators):
            return True

    return False


def is_bms_document(key: str, text: str = "") -> bool:
    """
    Determine if a document is a BMS document.
    Checks filename and content for BMS patterns.
    """
    key_lower = key.lower()

    # Check filename for BMS indicators
    bms_patterns = [
        "bms",
        "building management",
        "desops",
        "description of operation",
        "o&m",
        "operation & maintenance",
    ]

    if any(pattern in key_lower for pattern in bms_patterns):
        return True

    # Check content for BMS indicators (first and last sections)
    if text:
        text_sample = text[:3000] + (text[-1000:] if len(text) > 4000 else "")
        bms_indicators = [
            "building management system",
            "bms",
            "trend",
            "iq4",
            "controller",
            "ahu",
            "air handling unit",
            "hvac",
        ]
        text_lower = text_sample.lower()
        # Need at least 2 indicators to be confident
        matches = sum(
            1 for indicator in bms_indicators if indicator in text_lower)
        if matches >= 2:
            return True

    return False


def extract_text_csv_by_building_enhanced(
    key: str,
    data: bytes,
    alias_to_canonical: Dict[str, str]
) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """
    Extract CSV with enhanced metadata including aliases.

    Returns:
        List of (building_key, canonical_name, text, extra_metadata) tuples
    """
    try:
        df = pd.read_csv(io.BytesIO(data))

        building_col = "Property name"
        if building_col not in df.columns:
            logging.warning("Column '%s' not found. Available columns: %s",
                            building_col, df.columns.tolist())
            return [(key, "All Properties", df.to_string(), {})]

        building_docs = []
        for _, row in df.iterrows():
            building_name = row.get(building_col)
            if pd.isna(building_name):
                continue

            canonical_name = str(building_name).strip()

            # Build text representation
            building_text = f"Building: {canonical_name}\n\n"

            # Collect metadata for the building
            extra_metadata = {}

            for col, val in row.items():
                if pd.notna(val):
                    building_text += f"{col}: {val}\n"

                    # Store specific metadata fields
                    if col in ["Property names", "Property alternative names",
                               "UsrFRACondensedPropertyName", "Property code",
                               "Property postcode", "Property campus"]:
                        extra_metadata[col] = str(val)

            # Create searchable aliases list
            aliases = []

            # Add Property names
            if pd.notna(row.get("Property names")):
                aliases.extend([n.strip()
                               for n in str(row["Property names"]).split(";")])

            # Add alternative names
            if pd.notna(row.get("Property alternative names")):
                aliases.extend([n.strip() for n in str(
                    row["Property alternative names"]).split(";")])

            # Add condensed name
            if pd.notna(row.get("UsrFRACondensedPropertyName")):
                aliases.append(str(row["UsrFRACondensedPropertyName"]).strip())

            # Remove duplicates and empty strings
            aliases = list(set([a for a in aliases if a]))

            extra_metadata["building_aliases"] = aliases
            extra_metadata["canonical_building_name"] = canonical_name

            building_key = f"Planon Data - {canonical_name}"
            building_docs.append(
                (building_key, canonical_name, building_text, extra_metadata))

        if not building_docs:
            logging.warning(
                "No buildings found in CSV, indexing as single doc")
            return [(key, "All Properties", df.to_string(), {})]

        logging.info("Extracted %d buildings from CSV with aliases",
                     len(building_docs))
        return building_docs

    except Exception as ex:
        logging.warning("CSV extraction failed: %s", ex)
        return [(key, "", data.decode("utf-8", errors="ignore"), {})]


def extract_text(key: str, data: bytes) -> str:
    """Extract text from standard document formats (not CSV)."""
    e = ext(key)

    if e in {"txt", "md"}:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1", errors="ignore")
        return text

    if e == "json":
        try:
            text = data.decode("utf-8")
            text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        except Exception:  # pylint: disable=broad-except
            text = data.decode("utf-8", errors="ignore")
        return text

    if e == "pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n".join([p.extract_text() or "" for p in reader.pages])
        except Exception as ex:  # pylint: disable=broad-except
            logging.warning("PDF extract failed for %s: %s", key, ex)
            return ""

    if e == "docx":
        try:
            doc = DocxDocument(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as ex:  # pylint: disable=broad-except
            logging.warning("DOCX extract failed for %s: %s", key, ex)
            return ""

    return ""


def chunk_text(
    text: str, chunk_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP
) -> List[str]:
    """Split text into overlapping chunks based on token count."""
    toks = enc.encode(text)
    chunks = []
    i = 0
    while i < len(toks):
        j = min(i + chunk_tokens, len(toks))
        chunk = enc.decode(toks[i:j])
        chunk = re.sub(r"\s+\n", "\n", chunk).strip()
        if chunk:
            chunks.append(chunk)
        i = max(0, j - overlap)
        if j == len(toks):
            break
    return chunks


def embed_texts_batch(texts: List[str], max_retries: int = 5) -> List[List[float]]:
    """Embed a batch of texts with retry logic."""
    if not texts:
        return []
    for attempt in range(max_retries):
        try:
            resp = oai.embeddings.create(
                model=EMBED_MODEL,
                input=texts,
                timeout=OPENAI_TIMEOUT,
            )
            return [d.embedding for d in resp.data]
        except (RateLimitError, APIError) as e:
            logging.warning(
                "Embedding API backoff (attempt %d/%d): %s", attempt +
                1, max_retries, e
            )
            if attempt == max_retries - 1:
                return []
            backoff_sleep(attempt)
        except Exception as e:  # pylint: disable=broad-except
            logging.error("Embedding failed with unexpected error: %s", e)
            return []
    return []


def make_id(bucket: str, key: str, chunk_idx: int) -> str:
    """Generate a deterministic ID for a vector."""
    base = f"{bucket}:{key}:{chunk_idx}"
    return hashlib.md5(base.encode()).hexdigest()


def validate_metadata(metadata: dict) -> bool:
    """Ensure metadata has required fields."""
    required = ["bucket", "key", "source", "text", "document_type"]
    return all(k in metadata for k in required)


def upsert_vectors(vectors: List[Tuple[str, List[float], Dict]]):
    """Upsert vectors to Pinecone with retry logic."""
    for i in range(0, len(vectors), UPSERT_BATCH):
        batch = vectors[i: i + UPSERT_BATCH]
        for attempt in range(5):
            try:
                index.upsert(vectors=batch, namespace=NAMESPACE)
                stats["total_vectors"] += len(batch)
                break
            except Exception as e:  # pylint: disable=broad-except
                logging.warning(
                    "Pinecone upsert retry %d/5: %s", attempt + 1, e)
                if attempt == 4:
                    raise
                backoff_sleep(attempt)


# ---------------- Main ingest ----------------


def process_document(
    obj: Dict,
    bucket: str,
    name_to_canonical: Dict[str, str],
    alias_to_canonical: Dict[str, str],
    known_buildings: List[str],
    existing_vector_ids: Set[str],
) -> List[Tuple[str, List[float], Dict]]:
    """
    Process a single document and return vectors to upsert.
    Returns empty list if document should be skipped or fails.
    """
    key = obj["Key"]
    size = int(obj.get("Size") or 0)
    last_modified = obj.get("LastModified")
    size_mb = size / (1024 * 1024) if size else 0.0

    # Optional size guard
    if MAX_FILE_MB > 0 and size_mb > MAX_FILE_MB:
        logging.warning(
            "Skipping large file (%.1f MB > %.1f MB): %s", size_mb, MAX_FILE_MB, key
        )
        stats["files_skipped"] += 1
        return []

    logging.info("[1/5] Fetching: s3://%s/%s (size=%.2f MB)",
                 bucket, key, size_mb)
    t0 = time.time()

    try:
        data = fetch_bytes(bucket, key)
    except ClientError as e:
        logging.error("ERROR get_object %s: %s", key, e)
        stats["files_failed"] += 1
        stats["failed_files"].append(key)
        return []

    logging.info(
        "Fetched %d bytes in %.2fs; extracting...", len(data), time.time() - t0
    )

    vectors = []

    # ===== SPECIAL HANDLING FOR PROPERTY CSV =====
    if ext(key) == "csv" and "Property" in key:
        logging.info(
            "[2/5] Processing property CSV with building-specific extraction and aliases")
        t1 = time.time()
        building_docs = extract_text_csv_by_building_enhanced(
            key, data, alias_to_canonical
        )
        logging.info(
            "Extracted %d buildings in %.2fs", len(
                building_docs), time.time() - t1
        )

        for building_key, building_name, building_text, extra_metadata in building_docs:
            if not building_text.strip():
                continue

            t2 = time.time()
            chunks = chunk_text(building_text)
            logging.info(
                "[3/5] %s: %d chunks (chunk=%d, overlap=%d) in %.2fs",
                building_key,
                len(chunks),
                CHUNK_TOKENS,
                CHUNK_OVERLAP,
                time.time() - t2,
            )

            # Embed and create vectors
            for i in range(0, len(chunks), EMBED_BATCH):
                batch_chunks = chunks[i: i + EMBED_BATCH]

                # Check if chunks already exist
                batch_ids = [make_id(bucket, building_key, i + j)
                             for j in range(len(batch_chunks))]
                if SKIP_EXISTING and all(bid in existing_vector_ids for bid in batch_ids):
                    logging.info("Skipping %d existing chunks for %s",
                                 len(batch_chunks), building_key)
                    stats["vectors_skipped"] += len(batch_chunks)
                    continue

                logging.info(
                    "[4/5] Embedding chunks %d-%d of %d...",
                    i,
                    i + len(batch_chunks) - 1,
                    len(chunks),
                )
                t3 = time.time()
                embeddings = embed_texts_batch(batch_chunks)

                if not embeddings:
                    logging.error(
                        "Failed to embed chunks for %s", building_key)
                    stats["files_failed"] += 1
                    stats["failed_files"].append(building_key)
                    continue

                logging.info(
                    "Embedded %d chunks in %.2fs", len(
                        embeddings), time.time() - t3
                )

                for j, (emb, chunk_str) in enumerate(zip(embeddings, batch_chunks)):
                    cid = make_id(bucket, building_key, i + j)
                    metadata = {
                        "bucket": bucket,
                        "key": building_key,
                        "building_name": building_name,
                        "original_file": key,
                        "chunk": i + j,
                        "source": f"s3://{bucket}/{key}",
                        "size": size,
                        "last_modified": (last_modified.isoformat()
                                          if isinstance(last_modified, datetime)
                                          else str(last_modified)
                                          ),
                        "ext": "csv",
                        "text": chunk_str,
                        "document_type": "planon_data",
                        "is_building_specific": True,
                        "parent_document": key,
                        # Add alias metadata
                        "building_aliases": extra_metadata.get("building_aliases", []),
                        "canonical_building_name": building_name,
                    }

                    # Add other metadata fields
                    for field in ["Property code", "Property postcode", "Property campus",
                                  "UsrFRACondensedPropertyName", "Property names",
                                  "Property alternative names"]:
                        if field in extra_metadata:
                            metadata[field] = extra_metadata[field]

                    if validate_metadata(metadata):
                        vectors.append((cid, emb, metadata))

        stats["files_processed"] += 1
        logging.info("Finished CSV file: %s in %.2fs total",
                     key, time.time() - t0)
        return vectors

    # ===== STANDARD DOCUMENT PROCESSING (PDF, DOCX, etc.) =====
    t1 = time.time()
    text = extract_text(key, data)
    t_extract = time.time() - t1

    if not text.strip():
        logging.warning(
            "Skipping (no text): %s [extract %.2fs]", key, t_extract)
        stats["files_skipped"] += 1
        return []

    # Check for extremely large documents
    estimated = estimate_tokens(text)
    if estimated > 250000:  # ~1M characters
        logging.warning(
            "Document very large (~%d tokens), processing may be slow: %s",
            estimated,
            key,
        )

    logging.info(
        "[2/5] Extracted ~%d chars in %.2fs; chunking...", len(text), t_extract
    )

    # Determine document type
    is_fra = is_fire_risk_assessment(key, text)
    is_bms = is_bms_document(key, text)

    if is_fra:
        doc_type = "fire_risk_assessment"
    elif is_bms:
        doc_type = "operational_doc"
    else:
        doc_type = "unknown"

    # Extract building name from filename with enhanced fuzzy matching
    raw_building = extract_building_name_from_filename(key, known_buildings)
    building_name = find_closest_building_name_enhanced(
        raw_building,
        name_to_canonical,
        alias_to_canonical,
        known_buildings
    )

    logging.info("Building: '%s' (raw: '%s') - Type: %s",
                 building_name, raw_building, doc_type)

    t2 = time.time()
    chunks = chunk_text(text)
    logging.info(
        "[3/5] %s: %d chunks (chunk=%d, overlap=%d) in %.2fs",
        key,
        len(chunks),
        CHUNK_TOKENS,
        CHUNK_OVERLAP,
        time.time() - t2,
    )

    # Embed in batches
    for i in range(0, len(chunks), EMBED_BATCH):
        batch_chunks = chunks[i: i + EMBED_BATCH]

        # Check if chunks already exist
        batch_ids = [make_id(bucket, key, i + j)
                     for j in range(len(batch_chunks))]
        if SKIP_EXISTING and all(bid in existing_vector_ids for bid in batch_ids):
            logging.info("Skipping %d existing chunks for %s",
                         len(batch_chunks), key)
            stats["vectors_skipped"] += len(batch_chunks)
            continue

        logging.info(
            "[4/5] Embedding chunks %d-%d of %d...",
            i,
            i + len(batch_chunks) - 1,
            len(chunks),
        )
        t3 = time.time()
        embeddings = embed_texts_batch(batch_chunks)

        if not embeddings:
            logging.error("Failed to embed chunks for %s", key)
            stats["files_failed"] += 1
            stats["failed_files"].append(key)
            continue

        logging.info("Embedded %d chunks in %.2fs",
                     len(embeddings), time.time() - t3)

        for j, (emb, chunk_str) in enumerate(zip(embeddings, batch_chunks)):
            cid = make_id(bucket, key, i + j)
            metadata = {
                "bucket": bucket,
                "key": key,
                "building_name": building_name,
                "canonical_building_name": building_name,
                "original_file": key,
                "chunk": i + j,
                "source": f"s3://{bucket}/{key}",
                "size": size,
                "last_modified": (
                    last_modified.isoformat()
                    if isinstance(last_modified, datetime)
                    else str(last_modified)
                ),
                "ext": ext(key),
                "text": chunk_str,
                "document_type": doc_type,
            }

            if validate_metadata(metadata):
                vectors.append((cid, emb, metadata))

    stats["files_processed"] += 1
    logging.info("Finished file: %s in %.2fs total", key, time.time() - t0)
    return vectors


def ingest_bucket(bucket: str, prefix: str = ""):
    """Ingest documents from S3 bucket/prefix into Pinecone index."""
    t_start = time.time()
    objs = list_s3_objects(bucket, prefix)
    logging.info("Found %d files under s3://%s/%s", len(objs), bucket, prefix)

    # Get existing IDs to skip re-processing
    existing_vector_ids = get_existing_ids(bucket)

    # Pre-load building names with aliases from CSV if available
    name_to_canonical = {}
    alias_to_canonical = {}
    known_buildings = []

    csv_key = None
    for o in objs:
        if "Property" in o["Key"] and o["Key"].endswith(".csv"):
            csv_key = o["Key"]
            known_buildings, name_to_canonical, alias_to_canonical = \
                load_building_names_with_aliases(bucket, csv_key)
            break

    if not known_buildings:
        logging.warning(
            "No Property CSV found - building name matching may be limited")

    pending: List[Tuple[str, List[float], Dict]] = []

    # Process files (with optional parallelisation)
    if MAX_WORKERS > 1:
        logging.info("Processing files with %d workers...", MAX_WORKERS)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    process_document,
                    obj,
                    bucket,
                    name_to_canonical,
                    alias_to_canonical,
                    known_buildings,
                    existing_vector_ids
                ): obj
                for obj in objs
            }

            for future in as_completed(futures):
                obj = futures[future]
                try:
                    vectors = future.result()
                    pending.extend(vectors)

                    # Upsert when batch is ready
                    if len(pending) >= min(UPSERT_BATCH, MAX_PENDING_VECTORS):
                        logging.info(
                            "[5/5] Upserting %d vectors...", len(pending))
                        t4 = time.time()
                        upsert_vectors(pending)
                        logging.info("Upserted in %.2fs", time.time() - t4)
                        pending.clear()

                except Exception as e:  # pylint: disable=broad-except
                    logging.error(
                        "Error processing %s: %s", obj["Key"], e, exc_info=True
                    )
                    stats["files_failed"] += 1
                    stats["failed_files"].append(obj["Key"])
    else:
        # Sequential processing
        for obj in objs:
            vectors = process_document(
                obj,
                bucket,
                name_to_canonical,
                alias_to_canonical,
                known_buildings,
                existing_vector_ids
            )
            pending.extend(vectors)

            # Upsert when batch is ready
            if len(pending) >= min(UPSERT_BATCH, MAX_PENDING_VECTORS):
                logging.info("[5/5] Upserting %d vectors...", len(pending))
                t4 = time.time()
                upsert_vectors(pending)
                logging.info("Upserted in %.2fs", time.time() - t4)
                pending.clear()

    # Final upsert of remaining vectors
    if pending:
        logging.info("Final upsert of remaining %d vectors...", len(pending))
        t4 = time.time()
        upsert_vectors(pending)
        logging.info("Final upsert done in %.2fs", time.time() - t4)
        pending.clear()

    # Print statistics
    duration = time.time() - t_start
    vectors_per_sec = stats["total_vectors"] / duration if duration > 0 else 0

    logging.info(
        """
========================================
INGESTION SUMMARY
========================================
Files found:          %d
Files processed:      %d
Files skipped:        %d
Files failed:         %d
Total vectors:        %d
Vectors skipped:      %d
Duration:             %.2fs
Avg speed:            %.1f vectors/sec
========================================
""",
        len(objs),
        stats["files_processed"],
        stats["files_skipped"],
        stats["files_failed"],
        stats["total_vectors"],
        stats["vectors_skipped"],
        duration,
        vectors_per_sec,
    )

    if stats["failed_files"]:
        logging.warning("Failed files:")
        for f in stats["failed_files"]:
            logging.warning("  - %s", f)

    logging.info("Ingestion complete!")

# ---------------- CLI ----------------


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Ingest S3 docs into Pinecone via OpenAI embeddings"
    )
    p.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3 bucket name")
    p.add_argument(
        "--prefix", default=DEFAULT_PREFIX, help="S3 key prefix (folder)"
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=SKIP_EXISTING,
        help="Skip documents that already exist in the index",
    )
    p.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force re-indexing of all documents (overrides skip-existing)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help="Number of parallel workers for processing",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Override settings from CLI
    if args.force_reindex:
        SKIP_EXISTING = False
        logging.info("Force reindex enabled - will process all documents")
    elif args.skip_existing:
        SKIP_EXISTING = True
        logging.info(
            "Skip existing enabled - will skip already indexed documents")

    if args.workers:
        MAX_WORKERS = args.workers
        logging.info("Using %d workers for parallel processing", MAX_WORKERS)

    ingest_bucket(args.bucket, args.prefix)
