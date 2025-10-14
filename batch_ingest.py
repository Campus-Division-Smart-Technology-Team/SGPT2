import os
import io
import json
import re
import hashlib
import time
import math
import random
import argparse
import logging
from typing import Iterable, List, Dict, Tuple
from datetime import datetime
from difflib import get_close_matches

import boto3
from botocore.exceptions import ClientError
from botocore.config import Config as BotoConfig
from pypdf import PdfReader
from docx import Document as DocxDocument
import pandas as pd
import tiktoken
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
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
INDEX_NAME = os.getenv("INDEX_NAME", "bms")
NAMESPACE = os.getenv("NAMESPACE") or None
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
DIMENSION = int(os.getenv("DIMENSION", "1536"))
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))
UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", "64"))
MAX_FILE_MB = float(os.getenv("MAX_FILE_MB", "0"))
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "30"))

EXT_WHITELIST = {"txt", "md", "csv", "json", "pdf", "docx"}

# Global cache for building names
KNOWN_BUILDINGS_CACHE = []

# ---------------- Clients ----------------
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    config=BotoConfig(
        connect_timeout=5,
        read_timeout=60,
        retries={"max_attempts": 5}
    ),
)
pc = Pinecone(api_key=PINECONE_API_KEY)
oai = OpenAI(api_key=OPENAI_API_KEY)
enc = tiktoken.get_encoding("cl100k_base")

# Ensure index exists
existing = {i["name"] for i in pc.list_indexes()}
if INDEX_NAME not in existing:
    logging.info(f"Creating Pinecone index '{INDEX_NAME}' (dim={DIMENSION})…")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(INDEX_NAME)

# ---------------- Helpers ----------------


def backoff_sleep(attempt: int, base: float = 0.5, cap: float = 10.0):
    t = min(cap, base * (2 ** attempt)) * (0.5 + random.random())
    time.sleep(t)


def list_s3_objects(bucket: str, prefix: str = "") -> List[Dict]:
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
    r = s3.get_object(Bucket=bucket, Key=key)
    return r["Body"].read()


def ext(key: str) -> str:
    return key.lower().rsplit(".", 1)[-1] if "." in key else ""


def load_building_names_from_csv(bucket: str, key: str) -> List[str]:
    """Load all building names from the Property CSV for fuzzy matching."""
    global KNOWN_BUILDINGS_CACHE

    if KNOWN_BUILDINGS_CACHE:
        return KNOWN_BUILDINGS_CACHE

    try:
        data = fetch_bytes(bucket, key)
        df = pd.read_csv(io.BytesIO(data))

        if 'Property name' in df.columns:
            buildings = df['Property name'].dropna().unique().tolist()
            KNOWN_BUILDINGS_CACHE = buildings
            logging.info(
                f"Loaded {len(buildings)} building names for fuzzy matching")
            return buildings
    except Exception as e:
        logging.warning(
            f"Could not load building names for fuzzy matching: {e}")

    return []


def find_closest_building_name(extracted_name: str, known_buildings: List[str]) -> str:
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

    # Strategy 4: Use difflib for fuzzy match (60% similarity threshold)
    matches = get_close_matches(
        extracted_name, known_buildings, n=1, cutoff=0.6)
    if matches:
        return matches[0]

    # Return original if no good match found
    return extracted_name


def extract_building_name_from_filename(filename: str, known_buildings: List[str] = None) -> str:
    """
    Extract building name from PDF/DOCX filename with improved accuracy.
    Removes operational suffixes and optionally uses fuzzy matching.

    Example: 'UoB-Senate-House-BMS-Controls-Basement-Panel.pdf' -> 'Senate House'
    """
    # Remove path and extension
    name = os.path.basename(filename)
    name = name.replace('.pdf', '').replace('.docx', '').replace('.doc', '')

    # Remove UoB prefix
    name = name.replace('UoB-', '')

    # Generic patterns: remove common technical terms and suffixes
    patterns_to_remove = [
        r'-BMS.*$',              # Remove anything after -BMS
        r'-Controls.*$',         # Remove anything after -Controls
        r'-O&M.*$',              # Remove O&M manual references
        r'-OM.*$',               # Remove OM manual references
        r'-Des-Ops.*$',          # Remove Des-Ops references
        r'-Prototype.*$',        # Remove prototype lab references
        r'-Manual.*$',           # Remove manual references
        r'-Rev\d+.*$',           # Remove revision numbers (Rev1, Rev2, etc.)
        r'-rev\d+.*$',           # Remove lowercase revision numbers
        r'-P\d+.*$',             # Remove P numbers (P7391, etc.)
        r'-As-Installed.*$',     # Remove As-Installed suffix
        r'-Project.*$',          # Remove Project suffix
        r'-iiq.*$',              # Remove iiq suffix
        r'-trend.*$',            # Remove trend suffix
        r'-AC-to.*$',            # Remove AC-to prefix
    ]

    for pattern in patterns_to_remove:
        name = re.sub(pattern, '', name, flags=re.IGNORECASE)

    # Replace hyphens with spaces
    name = name.replace('-', ' ')

    # Clean up multiple spaces and trim
    name = re.sub(r'\s+', ' ', name).strip()

    # Capitalize properly (handles cases like "senate house" -> "Senate House")
    name = ' '.join(word.capitalize() for word in name.split())

    # If we have known buildings, try to match using fuzzy matching
    if known_buildings:
        matched_name = find_closest_building_name(name, known_buildings)
        if matched_name != name:
            logging.info(f"Fuzzy matched '{name}' -> '{matched_name}'")
            return matched_name

    return name


def extract_text_csv_by_building(key: str, data: bytes) -> List[Tuple[str, str, str]]:
    """
    Extract CSV and return list of (building_key, building_name, text) tuples.
    Each building becomes its own searchable document.
    """
    try:
        df = pd.read_csv(io.BytesIO(data))

        # Identify the building identifier column
        building_col = 'Property name'

        if building_col not in df.columns:
            logging.warning(
                f"Column '{building_col}' not found. Available columns: {df.columns.tolist()}")
            return [(key, "", df.to_string())]

        building_docs = []
        for idx, row in df.iterrows():
            building_name = row[building_col]
            if pd.isna(building_name):
                continue

            # Create a text representation of this building's data
            building_text = f"Building: {building_name}\n\n"
            for col, val in row.items():
                if pd.notna(val):
                    building_text += f"{col}: {val}\n"

            # Use building name as part of the key
            building_key = f"Planon Data - {building_name}"
            building_docs.append((building_key, building_name, building_text))

        logging.info(f"Extracted {len(building_docs)} buildings from CSV")
        return building_docs

    except Exception as ex:
        logging.warning(f"CSV building extraction failed for {key}: {ex}")
        return [(key, "", data.decode('utf-8', errors='ignore'))]


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
        except Exception:
            text = data.decode("utf-8", errors="ignore")
        return text

    if e == "pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n".join([p.extract_text() or "" for p in reader.pages])
        except Exception as ex:
            logging.warning(f"PDF extract failed for {key}: {ex}")
            return ""

    if e == "docx":
        try:
            doc = DocxDocument(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as ex:
            logging.warning(f"DOCX extract failed for {key}: {ex}")
            return ""

    return ""


def chunk_text(text: str, chunk_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP) -> List[str]:
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
                f"Embedding API backoff (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise
            backoff_sleep(attempt)
        except Exception as e:
            logging.error(f"Embedding failed with unexpected error: {e}")
            raise


def make_id(bucket: str, key: str, chunk_idx: int) -> str:
    base = f"{bucket}:{key}:{chunk_idx}"
    return hashlib.md5(base.encode()).hexdigest()


def upsert_vectors(vectors: List[Tuple[str, List[float], Dict]]):
    for i in range(0, len(vectors), UPSERT_BATCH):
        batch = vectors[i:i+UPSERT_BATCH]
        for attempt in range(5):
            try:
                index.upsert(vectors=batch, namespace=NAMESPACE)
                break
            except Exception as e:
                logging.warning(f"Pinecone upsert retry {attempt+1}/5: {e}")
                if attempt == 4:
                    raise
                backoff_sleep(attempt)


def test_building_name_extraction(known_buildings: List[str] = None):
    """Test building name extraction with sample filenames."""
    test_cases = [
        ("UoB-Senate-House-BMS-Controls-Basement-Panel.pdf", "Senate House"),
        ("UoB-Berkeley-Square.pdf", "Berkeley Square"),
        ("UoB-Dentistry-BMS-P7391-OM-As-Installed.pdf", "Dentistry"),
        ("UoB-Retort-House-BMS-O&M-Manual-rev1.pdf", "Retort House"),
        ("UoB-Whiteladies-Project-BMS-O&M-Manual.pdf", "Whiteladies"),
        ("UoB-Indoor-Sports-Hall.docx", "Indoor Sports Hall"),
        ("mitsubishi-ac-to-trend-iiq-interface.pdf",
         "Mitsubishi Ac To Trend Iiq Interface"),
    ]

    logging.info("=" * 60)
    logging.info("Testing building name extraction:")
    logging.info("=" * 60)

    for filename, expected in test_cases:
        result = extract_building_name_from_filename(filename, known_buildings)
        status = "✓" if result == expected else "✗"
        logging.info(f"{status} {filename}")
        logging.info(f"  Result: '{result}' | Expected: '{expected}'")
        if result != expected:
            logging.info(f"  MISMATCH!")
        logging.info("")

    logging.info("=" * 60)


# ---------------- Main ingest ----------------


def ingest_bucket(bucket: str, prefix: str = ""):
    t_start = time.time()
    objs = list_s3_objects(bucket, prefix)
    logging.info(f"Found {len(objs)} files under s3://{bucket}/{prefix}")

    # Pre-load building names from CSV if available
    known_buildings = []
    csv_key = None
    for o in objs:
        if 'Property' in o['Key'] and o['Key'].endswith('.csv'):
            csv_key = o['Key']
            known_buildings = load_building_names_from_csv(bucket, csv_key)
            break

    # Run test if buildings were loaded
    if known_buildings:
        test_building_name_extraction(known_buildings)

    pending: List[Tuple[str, List[float], Dict]] = []

    for o in objs:
        key = o["Key"]
        size = int(o.get("Size") or 0)
        last_modified = o.get("LastModified")
        size_mb = size / (1024 * 1024) if size else 0.0

        # optional size guard
        if MAX_FILE_MB > 0 and size_mb > MAX_FILE_MB:
            logging.warning(
                f"Skipping large file ({size_mb:.1f} MB > {MAX_FILE_MB} MB): {key}")
            continue

        logging.info(
            f"[1/5] Fetching: s3://{bucket}/{key} (size={size_mb:.2f} MB)")
        t0 = time.time()
        try:
            data = fetch_bytes(bucket, key)
        except ClientError as e:
            logging.error(f"ERROR get_object {key}: {e}")
            continue
        logging.info(
            f"Fetched {len(data)} bytes in {time.time()-t0:.2f}s; extracting…")

        # ===== SPECIAL HANDLING FOR PROPERTY CSV =====
        if ext(key) == "csv" and "Property" in key:
            logging.info(
                f"[2/5] Processing property CSV with building-specific extraction")
            t1 = time.time()
            building_docs = extract_text_csv_by_building(key, data)
            logging.info(
                f"Extracted {len(building_docs)} buildings in {time.time()-t1:.2f}s")

            for building_key, building_name, building_text in building_docs:
                if not building_text.strip():
                    continue

                t2 = time.time()
                chunks = chunk_text(building_text)
                logging.info(
                    f"[3/5] {building_key}: {len(chunks)} chunks (chunk={CHUNK_TOKENS}, overlap={CHUNK_OVERLAP}) in {time.time()-t2:.2f}s")

                # Embed and upsert with building-specific key
                for i in range(0, len(chunks), EMBED_BATCH):
                    batch_chunks = chunks[i:i+EMBED_BATCH]
                    logging.info(
                        f"[4/5] Embedding chunks {i}-{i+len(batch_chunks)-1} of {len(chunks)}…")
                    t3 = time.time()
                    embeddings = embed_texts_batch(batch_chunks)
                    logging.info(
                        f"Embedded {len(embeddings)} chunks in {time.time()-t3:.2f}s")

                    for j, (emb, chunk_str) in enumerate(zip(embeddings, batch_chunks)):
                        cid = make_id(bucket, building_key, i + j)
                        metadata = {
                            "bucket": bucket,
                            "key": building_key,  # Building-specific key
                            "building_name": building_name,  # Searchable field
                            "original_file": key,  # Reference to source CSV
                            "chunk": i + j,
                            "source": f"s3://{bucket}/{key}",
                            "size": size,
                            "last_modified": last_modified.isoformat() if isinstance(last_modified, datetime) else str(last_modified),
                            "ext": "csv",
                            "text": chunk_str,
                            "document_type": "planon_data"  # Tag for filtering
                        }
                        pending.append((cid, emb, metadata))

                    if len(pending) >= UPSERT_BATCH:
                        logging.info(
                            f"[5/5] Upserting {len(pending)} vectors…")
                        t4 = time.time()
                        upsert_vectors(pending)
                        logging.info(f"Upserted in {time.time()-t4:.2f}s")
                        pending.clear()

                logging.info(
                    f"Finished building: {building_key} in {time.time()-t2:.2f}s")

            logging.info(
                f"Finished CSV file: {key} in {time.time()-t0:.2f}s total")
            continue  # Skip to next file

        # ===== STANDARD DOCUMENT PROCESSING (PDF, DOCX, etc.) =====
        t1 = time.time()
        text = extract_text(key, data)
        t_extract = time.time() - t1
        if not text.strip():
            logging.warning(
                f"Skipping (no text): {key} [extract {t_extract:.2f}s]")
            continue
        logging.info(
            f"[2/5] Extracted ~{len(text)} chars in {t_extract:.2f}s; chunking…")

        t2 = time.time()
        chunks = chunk_text(text)
        logging.info(
            f"[3/5] {key}: {len(chunks)} chunks (chunk={CHUNK_TOKENS}, overlap={CHUNK_OVERLAP}) in {time.time()-t2:.2f}s")

        # Extract building name from filename with fuzzy matching
        building_name = extract_building_name_from_filename(
            key, known_buildings)
        logging.info(f"Extracted building name: '{building_name}' from {key}")

        # embed in batches
        for i in range(0, len(chunks), EMBED_BATCH):
            batch_chunks = chunks[i:i+EMBED_BATCH]
            logging.info(
                f"[4/5] Embedding chunks {i}-{i+len(batch_chunks)-1} of {len(chunks)}…")
            t3 = time.time()
            embeddings = embed_texts_batch(batch_chunks)
            logging.info(
                f"Embedded {len(embeddings)} chunks in {time.time()-t3:.2f}s")

            for j, (emb, chunk_str) in enumerate(zip(embeddings, batch_chunks)):
                cid = make_id(bucket, key, i + j)
                metadata = {
                    "bucket": bucket,
                    "key": key,
                    "building_name": building_name,  # Add building name for cross-reference
                    "original_file": key,
                    "chunk": i + j,
                    "source": f"s3://{bucket}/{key}",
                    "size": size,
                    "last_modified": last_modified.isoformat() if isinstance(last_modified, datetime) else str(last_modified),
                    "ext": ext(key),
                    "text": chunk_str,
                    "document_type": "operational_doc"  # Tag for filtering
                }
                pending.append((cid, emb, metadata))

            if len(pending) >= UPSERT_BATCH:
                logging.info(f"[5/5] Upserting {len(pending)} vectors…")
                t4 = time.time()
                upsert_vectors(pending)
                logging.info(f"Upserted in {time.time()-t4:.2f}s")
                pending.clear()

        logging.info(f"Finished file: {key} in {time.time()-t0:.2f}s total")

    if pending:
        logging.info(f"Final upsert of remaining {len(pending)} vectors…")
        t4 = time.time()
        upsert_vectors(pending)
        logging.info(f"Final upsert done in {time.time()-t4:.2f}s")
        pending.clear()

    logging.info(f"Ingestion complete in {time.time()-t_start:.2f}s.")


# ---------------- CLI ----------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Ingest S3 docs into Pinecone via OpenAI embeddings")
    p.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3 bucket name")
    p.add_argument("--prefix", default=DEFAULT_PREFIX,
                   help="S3 key prefix (folder)")
    p.add_argument("--test-extraction", action="store_true",
                   help="Run building name extraction tests only")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.test_extraction:
        # Just run the tests without ingestion
        logging.info("Running extraction tests only...")
        test_building_name_extraction()
    else:
        ingest_bucket(args.bucket, args.prefix)
