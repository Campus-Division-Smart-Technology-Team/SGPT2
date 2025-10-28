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
from pathlib import Path

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

# AWS credentials removed - not needed for local files
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

# -------------- Config (edit or override via env/CLI) --------------
# Changed to use local directory path
DEFAULT_LOCAL_PATH = os.getenv(
    "LOCAL_PATH", r"C:\Users\rd23091\Downloads\Alfred")
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
BUILDING_NAME_CACHE = {}  # Maps normalised names to canonical names
BUILDING_ALIASES_CACHE = {}  # Maps aliases to canonical names

# NEW: Global cache for building metadata from CSV
# Maps canonical building name to full metadata dict including aliases
BUILDING_METADATA_CACHE: Dict[str, Dict[str, Any]] = {}

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
# S3 client removed - not needed for local files
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


def backoff_sleep(attempt: int, base: float = 0.5, cap: float = 10.0) -> None:
    """Exponential backoff with jitter."""
    t = min(cap, base * (2**attempt)) * (0.5 + random.random())
    time.sleep(t)


def get_existing_ids(source_path: str) -> Set[str]:
    """
    Fetch all existing vector IDs for this source path to avoid re-embedding.
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


def list_local_files(base_path: str) -> List[Dict[str, Any]]:
    """
    List all files in a local directory recursively, filtered by extension.
    Returns list of dicts similar to S3 object format for compatibility.
    """
    base_path_obj = Path(base_path)

    if not base_path_obj.exists():
        raise FileNotFoundError(f"Directory not found: {base_path}")

    if not base_path_obj.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {base_path}")

    out = []

    # Recursively find all files
    for file_path in base_path_obj.rglob('*'):
        if file_path.is_file():
            file_ext = file_path.suffix.lstrip('.').lower()

            # Filter by extension
            if file_ext not in EXT_WHITELIST:
                continue

            # Get file stats
            stat = file_path.stat()

            # Create dict similar to S3 object format
            # Use relative path from base as "Key"
            relative_path = file_path.relative_to(base_path_obj)
            # Normalize to forward slashes
            key = str(relative_path).replace('\\', '/')

            obj = {
                "Key": key,
                "Size": stat.st_size,
                "LastModified": datetime.fromtimestamp(stat.st_mtime),
                "FullPath": str(file_path)  # Store full path for easy access
            }
            out.append(obj)

    return out


def fetch_bytes(base_path: str, key: str) -> bytes:
    """Fetch file bytes from local filesystem."""
    # Reconstruct full path
    file_path = Path(base_path) / key

    try:
        with open(file_path, 'rb') as f:
            return f.read()
    except Exception as e:
        logging.error("Error reading file %s: %s", file_path, e)
        raise


def ext(key: str) -> str:
    """Get file extension from file path."""
    return key.lower().rsplit(".", 1)[-1] if "." in key else ""


def estimate_tokens(text: str) -> int:
    """Fast token estimation: ~4 chars per token."""
    return len(text) // 4


def load_building_names_with_aliases(base_path: str, key: str) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """
    Load building names and create comprehensive alias mappings from local CSV.

    Returns:
        (canonical_names, name_to_canonical_map, alias_to_canonical_map)
    """
    global BUILDING_NAME_CACHE, BUILDING_ALIASES_CACHE

    if BUILDING_NAME_CACHE and BUILDING_ALIASES_CACHE:
        return list(set(BUILDING_NAME_CACHE.values())), BUILDING_NAME_CACHE, BUILDING_ALIASES_CACHE

    try:
        data = fetch_bytes(base_path, key)
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

            # Map normalised primary name to canonical
            normalised_primary = primary_name.lower().strip()
            name_to_canonical[normalised_primary] = primary_name
            # FIXED: Ensure canonical maps to itself
            alias_to_canonical[normalised_primary] = primary_name
            # Ensure canonical name maps to itself
            alias_to_canonical[normalised_primary] = primary_name

            # Process Property names (comma-separated)
            property_names = row.get("Property names")
            if pd.notna(property_names):
                for name in str(property_names).split(","):
                    name = name.strip()
                    if name:
                        normalised = name.lower().strip()
                        alias_to_canonical[normalised] = primary_name
                        name_to_canonical[normalised] = primary_name

            # Process Property alternative names (comma-separated)
            alt_names = row.get("Property alternative names")
            if pd.notna(alt_names):
                for name in str(alt_names).split(","):
                    name = name.strip()
                    if name:
                        normalised = name.lower().strip()
                        alias_to_canonical[normalised] = primary_name

            # Process UsrFRACondensedPropertyName (common abbreviations)
            condensed = row.get("UsrFRACondensedPropertyName")
            if pd.notna(condensed):
                condensed = str(condensed).strip()
                if condensed:
                    normalised = condensed.lower().strip()
                    alias_to_canonical[normalised] = primary_name

            # NEW: Collect all aliases for metadata cache
            aliases = []

            # Add Property names
            if pd.notna(property_names):
                aliases.extend([n.strip() for n in str(
                    property_names).split(",") if n.strip()])

            # Add alternative names
            if pd.notna(alt_names):
                aliases.extend([n.strip()
                               for n in str(alt_names).split(",") if n.strip()])

            # Add condensed name
            if pd.notna(condensed):
                aliases.append(condensed)

            # Remove duplicates while preserving order
            unique_aliases = []
            seen = set()
            for alias in aliases:
                alias_lower = alias.lower()
                if alias_lower not in seen:
                    seen.add(alias_lower)
                    unique_aliases.append(alias)

            # NEW: Store complete metadata for this building
            building_metadata = {
                "canonical_building_name": primary_name,
                "building_aliases": unique_aliases,
            }

            # Add other metadata fields
            for field in ["Property code", "Property postcode", "Property campus",
                          "UsrFRACondensedPropertyName", "Property names",
                          "Property alternative names"]:
                if pd.notna(row.get(field)):
                    building_metadata[field] = str(row[field])

            # Store in cache with both canonical and normalised keys
            BUILDING_METADATA_CACHE[primary_name] = building_metadata
            BUILDING_METADATA_CACHE[normalised_primary] = building_metadata

        # Update global caches
        BUILDING_NAME_CACHE = name_to_canonical
        BUILDING_ALIASES_CACHE = alias_to_canonical

        logging.info(
            "Loaded %d canonical building names with %d aliases",
            len(canonical_names),
            len(alias_to_canonical)
        )
        logging.info(
            "✅ Populated BUILDING_METADATA_CACHE with metadata for %d buildings",
            # Count only canonical names (non-lowercase keys)
            len([k for k in BUILDING_METADATA_CACHE.keys() if not k.islower()])
        )

        return canonical_names, name_to_canonical, alias_to_canonical

    except Exception as e:  # pylint: disable=broad-except
        logging.warning("Could not load building names with aliases: %s", e)
        return [], {}, {}


def get_building_metadata(building_name: str) -> Dict[str, Any]:
    """
    Get full building metadata including aliases from cache.

    This function looks up a building name (canonical or alias) and returns
    the complete metadata including building_aliases that were loaded from CSV.

    Args:
        building_name: Canonical building name or any known alias

    Returns:
        Dictionary with building metadata including aliases, or empty dict if not found
    """
    if not building_name:
        return {}

    # Try exact match first (canonical name)
    if building_name in BUILDING_METADATA_CACHE:
        return BUILDING_METADATA_CACHE[building_name].copy()

    # Try normalised lookup
    normalised = building_name.lower().strip()
    if normalised in BUILDING_METADATA_CACHE:
        return BUILDING_METADATA_CACHE[normalised].copy()

    # Try to resolve through alias cache to get canonical, then lookup metadata
    canonical = BUILDING_ALIASES_CACHE.get(normalised)
    if canonical and canonical in BUILDING_METADATA_CACHE:
        return BUILDING_METADATA_CACHE[canonical].copy()

    # Try normalised canonical from NAME_CACHE
    canonical = BUILDING_NAME_CACHE.get(normalised)
    if canonical and canonical in BUILDING_METADATA_CACHE:
        return BUILDING_METADATA_CACHE[canonical].copy()

    logging.debug("No metadata found in cache for building: %s", building_name)
    return {}


def find_closest_building_name(
    extracted_name: str, known_buildings: List[str]
) -> str:
    """
    Find the closest matching building name using fuzzy matching.
    IMPROVED: Requires substantial overlap for substring matches.
    """
    if not extracted_name or not known_buildings:
        return extracted_name

    # Strategy 1: Try exact match first (case-insensitive)
    for building in known_buildings:
        if building.lower() == extracted_name.lower():
            return building

    # Strategy 2: Try substring match with overlap requirement
    # (extracted name contained in building name)
    for building in known_buildings:
        if extracted_name.lower() in building.lower():
            # Require at least 70% overlap
            overlap_pct = len(extracted_name) / len(building)
            if overlap_pct >= 0.7:
                return building

    # Strategy 3: Try reverse substring match with overlap requirement
    # (building name contained in extracted name)
    for building in known_buildings:
        if building.lower() in extracted_name.lower():
            # Require at least 70% overlap
            overlap_pct = len(building) / len(extracted_name)
            if overlap_pct >= 0.7:
                return building

    # Strategy 4: Use difflib for fuzzy match (85% similarity threshold - increased from 80%)
    matches = get_close_matches(
        extracted_name, known_buildings, n=1, cutoff=0.85)
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
        name_to_canonical: Map of normalised names to canonical names
        alias_to_canonical: Map of aliases to canonical names
        known_buildings: List of canonical building names (for fallback)

    Returns:
        Matched canonical building name
    """
    if not extracted_name:
        return extracted_name

    normalised = extracted_name.lower().strip()

    # Strategy 1: Direct match in canonical names
    if normalised in name_to_canonical:
        canonical = name_to_canonical[normalised]
        logging.info("Exact match: '%s' -> '%s'", extracted_name, canonical)
        return canonical

    # Strategy 2: Match in aliases
    if normalised in alias_to_canonical:
        canonical = alias_to_canonical[normalised]
        logging.info("Alias match: '%s' -> '%s'", extracted_name, canonical)
        return canonical

    # Strategy 3: Substring match in canonical names (IMPROVED: require substantial overlap)
    for norm_name, canonical in name_to_canonical.items():
        # Require at least 70% of the shorter string to match
        shorter_len = min(len(normalised), len(norm_name))
        if shorter_len < 5:  # Skip very short strings to avoid false matches
            continue

        if normalised in norm_name or norm_name in normalised:
            # Calculate overlap percentage
            if normalised in norm_name:
                overlap_pct = len(normalised) / len(norm_name)
            else:
                overlap_pct = len(norm_name) / len(normalised)

            # Require significant overlap (70%)
            if overlap_pct >= 0.7:
                logging.info("Substring match: '%s' -> '%s' (overlap: %.1f%%)",
                             extracted_name, canonical, overlap_pct * 100)
                return canonical

    # Strategy 4: Substring match in aliases (IMPROVED: require substantial overlap)
    for alias, canonical in alias_to_canonical.items():
        # Require at least 70% of the shorter string to match
        shorter_len = min(len(normalised), len(alias))
        if shorter_len < 5:  # Skip very short strings
            continue

        if normalised in alias or alias in normalised:
            # Calculate overlap percentage
            if normalised in alias:
                overlap_pct = len(normalised) / len(alias)
            else:
                overlap_pct = len(alias) / len(normalised)

            # Require significant overlap (70%)
            if overlap_pct >= 0.7:
                logging.info("Alias substring match: '%s' -> '%s' (overlap: %.1f%%)",
                             extracted_name, canonical, overlap_pct * 100)
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

    Supported FRA patterns:
    - RFM-FRA-BuildingName-YYYY-MM
    - FM-OAS-BuildingName-YYYY-MM
    - FRA-## BuildingName-YYYY-MM
    - FM-FRA-BuildingName-YYYY-MM
    - SRL-FRA-BuildingName-YYYY-MM
    - UoB-BuildingName-BMS-...
    """
    name = os.path.basename(filename)
    name = re.sub(r"\.(pdf|docx?|DOCX?)$", "", name, flags=re.IGNORECASE)

    # Pattern 1: RFM-FRA-BuildingName-YYYY-MM (flexible: handles YYYY-M, YYYY, or space before year)
    if (rfm_match := re.match(r"RFM-FRA-(.+?)(?:-|\s+)(\d{4})(?:-\d{1,2})?$", name, re.IGNORECASE)):
        building_part = rfm_match.group(1)

    # Pattern 2: FM-OAS-BuildingName-YYYY-MM (flexible date formats)
    elif (oas_match := re.match(r"FM-OAS-(.+?)(?:-|\s+)(\d{4})(?:-\d{1,2})?$", name, re.IGNORECASE)):
        building_part = oas_match.group(1)

    # Pattern 3: FRA-## BuildingName-YYYY-MM (flexible date formats)
    elif (fra_num_match := re.match(r"FRA-\d+\s+(.+?)(?:-|\s+)(\d{4})(?:-\d{1,2})?$", name, re.IGNORECASE)):
        building_part = fra_num_match.group(1)

    # Pattern 4: FM-FRA-BuildingName-YYYY-MM (or optional FM- prefix, flexible date)
    elif (fra_match := re.match(r"(?:FM-)?FRA-(.+?)(?:-|\s+)(\d{4})(?:-\d{1,2})?$", name, re.IGNORECASE)):
        building_part = fra_match.group(1)

    # Pattern 5: SRL-FRA-BuildingName-YYYY-MM (flexible date formats)
    elif (srl_match := re.match(r"SRL-FRA-(.+?)(?:-|\s+)(\d{4})(?:-\d{1,2})?$", name, re.IGNORECASE)):
        building_part = srl_match.group(1)

    # Pattern 6: UoB-BuildingName-BMS
    elif (bms_match := re.match(r"UoB-(.+?)-BMS", name, re.IGNORECASE)):
        building_part = bms_match.group(1)

    else:
        building_part = name  # fallback if no pattern matched

    # --- Normalise building_part ---
    building_part = re.sub(r"([a-z])([A-Z])", r"\1 \2", building_part)
    building_part = re.sub(r"(\d)([A-Z])", r"\1 \2", building_part)

    # Remove UoB prefix if still present
    building_part = building_part.replace("UoB-", "")

    # Remove common suffix patterns (updated to handle spaces before FRA/SRL)
    patterns_to_remove = [
        r"-BMS.*$",
        r"-Controls.*$",
        r"-O&M.*$",
        r"-OM.*$",
        r"-Des-?Ops.*$",
        r"-Prototype.*$",
        r"-Manual.*$",
        r"-Rev\d+.*$",
        r"-P\d+.*$",
        r"-As-Installed.*$",
        r"-Project.*$",
        r"-iiq.*$",
        r"-trend.*$",
        r"-AC-to.*$",
        r"[\s-]FRA$",      # Match both "-FRA" and " FRA" at end of string
        r"[\s-]SRL$",      # Match both "-SRL" and " SRL" at end of string
        r"[\s-]\d{4}(?:-\d{1,2})?$",  # Match dates at end with flexible format
        r"-v\d+$",
        r"-draft$",
    ]
    for pattern in patterns_to_remove:
        building_part = re.sub(pattern, "", building_part, flags=re.IGNORECASE)

    # Clean and format final name
    # Preserve hyphens between numbers (e.g., "1-9") but replace other hyphens with spaces
    # First, temporarily protect number-to-number hyphens
    building_part = re.sub(r'(\d)-(\d)', r'\1HYPHEN\2', building_part)
    # Now replace remaining hyphens with spaces
    building_part = building_part.replace("-", " ")
    # Restore the protected hyphens
    building_part = building_part.replace("HYPHEN", "-")

    # Expand common abbreviations
    abbreviations = {
        r'\bRd\b': 'Road',
        r'\bAve\b': 'Avenue',
        r'\bDr\b': 'Drive',
        r'\bTPR\b': 'Tyndalls Park Road',
        r'\bWR\b': 'Woodland Road',
        r'\bSMP\b': 'St Michaels Park',
        r'\bSMH\b': 'St Michaels Hill',
        r'\bBSQ\b': 'Berkeley Square',
        r'\bAccommodation@(\d+)': r'Accommodation at \1',
        r'\bR[/\\]?O\b': 'Rear Of',  # Handle R/O, R\O, or RO
    }

    for abbr, expansion in abbreviations.items():
        building_part = re.sub(
            abbr, expansion, building_part, flags=re.IGNORECASE)

    # Handle consecutive standalone numbers (e.g., "34 35" -> "34-35")
    building_part = re.sub(
        r'\b(\d{1,2})\s+(\d{1,2})\s+(?![Rr]oad|[Pp]ark|[Hh]ill)', r'\1-\2 ', building_part)
    building_part = re.sub(
        r'(\d{1,2})\s+(\d{1,2})(?=\s+[A-Z]|$)', r'\1-\2', building_part)

    building_part = re.sub(r"\s+", " ", building_part).strip()

    # If result is too short or just a prefix (likely extraction error), log warning
    if len(building_part) <= 3 and building_part.upper() in ['FM', 'RFM', 'SRL', 'RFL', 'RFH', 'OAS']:
        logging.warning(
            f"⚠️  Invalid building name extracted: '{building_part}' from '{filename}' - using full filename as fallback")
        building_part = filename.replace('.pdf', '').replace(
            '.docx', '').replace('.DOCX', '')

    # Capitalize words, but preserve special cases
    words = building_part.split()
    capitalized_words = []
    for word in words:
        # Keep words with internal hyphens or @ symbols as-is, just capitalize first letter
        if '-' in word and not word[0].isdigit():
            capitalized_words.append(word.capitalize())
        elif '@' in word:
            parts = word.split('@')
            capitalized_words.append('@'.join(p.capitalize() for p in parts))
        else:
            capitalized_words.append(word.capitalize())

    name = " ".join(capitalized_words)

    return name


def is_fire_risk_assessment(key: str, text: str = "") -> bool:
    """
    Determine if a document is a Fire Risk Assessment.
    Checks filename and content for multiple patterns.

    Recognizes patterns:
    - RFM-FRA-* (Fire Risk Management - Fire Risk Assessment)
    - FM-FRA-* (Facilities Management - Fire Risk Assessment)
    - FM-OAS-* (Facilities Management - Operational Assessment Sheet)
    - FRA-## * (Fire Risk Assessment with number)
    - SRL-FRA-* (older checksheet style)
    """
    # Check filename patterns
    key_lower = key.lower()

    # Common FRA filename patterns
    fra_patterns = [
        "rfm-fra",      # New pattern: RFM-FRA-BuildingName
        "fm-oas",       # New pattern: FM-OAS-BuildingName
        "fra",          # Generic FRA
        "fire-risk",
        "fire_risk",
        "srl-fra",
        "fm-fra",
    ]

    if any(pattern in key_lower for pattern in fra_patterns):
        return True

    # Check for FRA-## pattern (e.g., FRA-65 ST MichaelsHill)
    if re.search(r'fra-\d+\s+', key_lower):
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
            "operational assessment",  # For OAS documents
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

    except Exception as ex:  # pylint: disable=broad-except
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


def make_id(source_path: str, key: str, chunk_idx: int) -> str:
    """Generate a deterministic ID for a vector."""
    base = f"{source_path}:{key}:{chunk_idx}"
    return hashlib.md5(base.encode()).hexdigest()


def validate_metadata(metadata: Dict[str, Any]) -> bool:
    """Ensure metadata has required fields."""
    required = ["source_path", "key", "source", "text", "document_type"]
    return all(k in metadata for k in required)


def upsert_vectors(vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> None:
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
    obj: Dict[str, Any],
    base_path: str,
    name_to_canonical: Dict[str, str],
    alias_to_canonical: Dict[str, str],
    known_buildings: List[str],
    existing_vector_ids: Set[str],
) -> List[Tuple[str, List[float], Dict[str, Any]]]:
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

    logging.info("[1/5] Fetching: %s (size=%.2f MB)",
                 key, size_mb)
    t0 = time.time()

    try:
        data = fetch_bytes(base_path, key)
    except Exception as e:  # pylint: disable=broad-except
        logging.error("ERROR reading file %s: %s", key, e)
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
                batch_ids = [make_id(base_path, building_key, i + j)
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
                    cid = make_id(base_path, building_key, i + j)
                    metadata = {
                        "source_path": base_path,
                        "key": building_key,
                        "building_name": building_name,
                        "original_file": key,
                        "chunk": i + j,
                        "source": f"local://{base_path}/{key}",
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
        batch_ids = [make_id(base_path, key, i + j)
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
            cid = make_id(base_path, key, i + j)
            metadata = {
                "source_path": base_path,
                "key": key,
                "building_name": building_name,
                "canonical_building_name": building_name,
                "original_file": key,
                "chunk": i + j,
                "source": f"local://{base_path}/{key}",
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

            # NEW: Add building metadata (including aliases) from cache
            # This ensures non-CSV documents have the same metadata fields as CSV documents
            cached_building_metadata = get_building_metadata(building_name)
            if cached_building_metadata:
                # Add building_aliases if available
                if "building_aliases" in cached_building_metadata:
                    metadata["building_aliases"] = cached_building_metadata["building_aliases"]
                    if i == 0 and j == 0:  # Log only once per file
                        logging.info(
                            "✅ Added %d building aliases from cache to %s: %s",
                            len(cached_building_metadata["building_aliases"]),
                            doc_type,
                            ", ".join(cached_building_metadata["building_aliases"][:3]) +
                            ("..." if len(
                                cached_building_metadata["building_aliases"]) > 3 else "")
                        )

                # Add other metadata fields from CSV for consistency
                for field in ["Property code", "Property postcode", "Property campus",
                              "UsrFRACondensedPropertyName", "Property names",
                              "Property alternative names"]:
                    if field in cached_building_metadata:
                        metadata[field] = cached_building_metadata[field]
            else:
                # If no match in cache, add empty building_aliases for consistency
                metadata["building_aliases"] = []
                if i == 0 and j == 0:  # Log only once per file
                    logging.debug(
                        "No cached metadata for building '%s' in %s, using empty aliases",
                        building_name,
                        doc_type
                    )

            if validate_metadata(metadata):
                vectors.append((cid, emb, metadata))

    stats["files_processed"] += 1
    logging.info("Finished file: %s in %.2fs total", key, time.time() - t0)
    return vectors


def ingest_local_directory(base_path: str) -> None:
    """Ingest documents from local directory into Pinecone index."""
    t_start = time.time()

    # Validate path exists
    if not Path(base_path).exists():
        raise FileNotFoundError(f"Directory not found: {base_path}")

    objs = list_local_files(base_path)
    logging.info("Found %d files in %s", len(objs), base_path)

    # Get existing IDs to skip re-processing
    existing_vector_ids = get_existing_ids(base_path)

    # Pre-load building names with aliases from CSV if available
    name_to_canonical = {}
    alias_to_canonical = {}
    known_buildings = []

    csv_key = None
    for o in objs:
        if "Property" in o["Key"] and o["Key"].endswith(".csv"):
            csv_key = o["Key"]
            known_buildings, name_to_canonical, alias_to_canonical = \
                load_building_names_with_aliases(base_path, csv_key)
            break

    if not known_buildings:
        logging.warning(
            "No Property CSV found - building name matching may be limited")

    pending: List[Tuple[str, List[float], Dict[str, Any]]] = []

    # Process files (with optional parallelisation)
    if MAX_WORKERS > 1:
        logging.info("Processing files with %d workers...", MAX_WORKERS)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    process_document,
                    obj,
                    base_path,
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
                base_path,
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
        description="Ingest local documents into Pinecone via OpenAI embeddings"
    )
    p.add_argument(
        "--path",
        default=DEFAULT_LOCAL_PATH,
        help=f"Local directory path (default: {DEFAULT_LOCAL_PATH})"
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

    ingest_local_directory(args.path)
