#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date parsing and publication date search utilities.
Optimized version with pre-compiled patterns and better validation.
"""

import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from pinecone_utils import normalise_matches
from config import DEFAULT_NAMESPACE, DIMENSION

# ============================================================================
# CONSTANTS
# ============================================================================

# Date validation parameters (configurable)
MIN_YEAR = 1990  # Earliest reasonable year for BMS documents
MAX_AGE_YEARS = 30  # Maximum age for documents to be considered valid
DAYS_PER_YEAR = 365.25  # Account for leap years

# Dummy vector for metadata-only queries
DUMMY_VECTOR = [0.0] * DIMENSION

# Metadata date fields in priority order (skip unreliable fields)
METADATA_DATE_FIELDS = ['review_date', 'updated',
                        'revised', 'date', 'document_date']

# Date format strings for parsing
DATE_FORMATS = [
    "%d %B %Y",      # 19 March 2020
    "%d.%m.%Y",      # 12.06.2025 (dot format)
    "%Y.%m.%d",      # 2025.06.12 (dot format)
    "%Y-%m-%d",      # 2025-07-28
    "%B %d, %Y",     # July 28, 2025
    "%d/%m/%Y",      # 28/07/2025
    "%m/%d/%Y",      # 07/28/2025
    "%Y",            # 2025
    "%B %Y",         # July 2025
    "%d-%m-%Y",      # 28-07-2025
    "%Y/%m/%d",      # 2025/07/28
    "%d %b %Y",      # 19 Mar 2020 (abbreviated month)
    "%d-%b-%Y",      # 19-Mar-2020
]

# Pre-compiled regex patterns for date extraction
# Format: (pattern, pattern_type, priority_score)
# Higher priority = more reliable date source

# Helper function to build common pattern groups


def _build_label_pattern(labels: List[str], date_format: str) -> str:
    """Build a regex pattern for labeled dates."""
    label_group = '|'.join(labels)
    return f'(?:{label_group})[:\s]+({date_format})'


# Date format components
DATE_TEXT = r'[0-3]?[0-9][\s][A-Z][a-z]+[\s][0-9]{{4}}'
DATE_TEXT_FLEX = r'[0-3]?[0-9][\s/][A-Za-z]{{3,9}}[\s/][0-9]{{4}}'
DATE_NUMERIC = r'[0-3]?[0-9][\s/.-][0-3]?[0-9][\s/.-][0-9]{{4}}'
DATE_ABBREV = r'[0-3]?[0-9][\s][A-Z][a-z]{{2}}[\s][0-9]{{4}}'
DATE_DOT_DMY = r'[0-3]?[0-9]\.[0-1]?[0-9]\.[0-9]{{4}}'
DATE_DOT_YMD = r'[0-9]{{4}}\.[0-1]?[0-9]\.[0-3]?[0-9]'
DATE_ISO = r'[0-9]{{4}}[/-][0-1]?[0-9][/-][0-3]?[0-9]'
YEAR_ONLY = r'[0-9]{{4}}'
DATE_TEXT_ORDINAL = r'[0-3]?[0-9](?:st|nd|rd|th)?[\s][A-Z][a-z]+[\s][0-9]{4}'

# Pre-compiled date patterns with priorities
DATE_PATTERNS = [
    # Highest priority: Explicitly labeled dates
    (re.compile(_build_label_pattern(['Last\s+Updated', 'Last\s+Revised', 'Date\s+Updated',
     'Date\s+Revised'], DATE_TEXT_ORDINAL), re.IGNORECASE), 'labeled_updated', 15),
    (re.compile(_build_label_pattern(['Last\s+Updated', 'Last\s+Revised', 'Date\s+Updated',
     'Date\s+Revised'], DATE_NUMERIC), re.IGNORECASE), 'labeled_updated_numeric', 15),

    # Document header patterns
    (re.compile(_build_label_pattern(['Document\s+Date', 'Issue\s+Date',
     'Effective\s+Date'], DATE_TEXT_ORDINAL), re.IGNORECASE), 'doc_header', 14),
    (re.compile(
        r'(?:Version|Rev\.?\s+|Revision)[:\s]*([0-9]{1,2}[\s/.-][A-Z][a-z]{2}[\s/.-][0-9]{4})', re.IGNORECASE), 'version_date', 13),

    # General labeled dates
    (re.compile(_build_label_pattern(['Updated', 'Revised', 'Review\s+Date', 'Publication\s+Date',
     'Published'], DATE_TEXT_ORDINAL), re.IGNORECASE), 'labeled_general', 12),
    (re.compile(_build_label_pattern(
        ['Updated', 'Revised'], DATE_NUMERIC), re.IGNORECASE), 'labeled_general_numeric', 12),

    # Standard text dates (with ordinals)
    (re.compile(rf'\b({DATE_TEXT_ORDINAL})\b',
     re.IGNORECASE), 'standard_text_ordinal', 10),
    (re.compile(rf'\b({DATE_TEXT})\b', re.IGNORECASE), 'standard_text', 10),
    (re.compile(rf'\b({DATE_DOT_DMY})\b', re.IGNORECASE), 'dot_dmy', 8),
    (re.compile(rf'\b({DATE_DOT_YMD})\b', re.IGNORECASE), 'dot_ymd', 8),
    (re.compile(rf'\b({DATE_TEXT_FLEX})\b',
     re.IGNORECASE), 'standard_text_slash', 7),
    (re.compile(rf'\b({DATE_ABBREV})\b',
     re.IGNORECASE), 'abbreviated_text', 7),

    # Numeric formats
    (re.compile(rf'\b({DATE_NUMERIC})\b',
     re.IGNORECASE), 'standard_numeric', 6),
    (re.compile(rf'\b({DATE_ISO})\b', re.IGNORECASE), 'iso_date', 6),

    # Low priority: Copyright/version years (often template dates)
    (re.compile(
        rf'(?:©|Copyright)[:\s]*({YEAR_ONLY})', re.IGNORECASE), 'copyright_year', 3),
    (re.compile(
        rf'(?:Rev\.?\s*|Version\s+)[:\s]*({YEAR_ONLY})', re.IGNORECASE), 'version_year', 3),
]

# Quick patterns for single-result extraction (pre-compiled)
QUICK_DATE_PATTERNS = [
    re.compile(
        r'(?:Last\s+Updated|Last\s+Revised|Updated|Revised|Published)[:\s]+([0-3]?[0-9](?:st|nd|rd|th)?[\s/.-][A-Za-z]+[\s/.-][0-9]{4})', re.IGNORECASE),
    re.compile(
        r'(?:Last\s+Updated|Last\s+Revised|Updated|Revised)[:\s]+([0-3]?[0-9][\s/.-][0-3]?[0-9][\s/.-][0-9]{4})', re.IGNORECASE),
    re.compile(rf'\b({DATE_TEXT_ORDINAL})\b', re.IGNORECASE),
    re.compile(rf'\b({DATE_TEXT})\b', re.IGNORECASE),
]


# ============================================================================
# DATE PARSING
# ============================================================================


def parse_date_string(date_str: str) -> datetime:
    """
    Parse various date formats and return a datetime object.
    Returns datetime.min if parsing fails.
    Handles ordinal suffixes (1st, 2nd, 3rd, etc.)
    """
    if not date_str or date_str == "publication date unknown":
        return datetime.min

    date_str = date_str.strip()

    # Handle ordinal suffixes (1st, 2nd, 3rd, 21st, etc.)
    # Remove ordinal suffixes before parsing
    date_str_clean = re.sub(r'\b(\d{1,2})(st|nd|rd|th)\b', r'\1', date_str)

    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str_clean, fmt)
        except ValueError:
            continue

    # If no format matches, return minimum date
    return datetime.min


def format_display_date(date_obj: datetime) -> str:
    """Format datetime object for user-friendly display."""
    if date_obj == datetime.min:
        return "Unknown"

    if date_obj.day == 1 and date_obj.month == 1:
        # Year only
        return date_obj.strftime("%Y")
    elif date_obj.day == 1:
        # Month and year
        return date_obj.strftime("%B %Y")
    else:
        # Full date
        return date_obj.strftime("%d %B %Y")


def is_valid_date(parsed_date: datetime) -> bool:
    """
    Validate that a parsed date is reasonable for a technical document.

    Args:
        parsed_date: Parsed datetime object

    Returns:
        True if date is valid and reasonable
    """
    if parsed_date == datetime.min:
        return False

    current_date = datetime.now()

    # Date must not be in the future
    if parsed_date > current_date:
        return False

    # Date must not be older than MAX_AGE_YEARS
    age_days = (current_date - parsed_date).days
    if age_days > MAX_AGE_YEARS * DAYS_PER_YEAR:
        return False

    # Date must not be before MIN_YEAR (unlikely for BMS docs)
    if parsed_date.year < MIN_YEAR:
        return False

    return True


# ============================================================================
# DATE EXTRACTION FROM METADATA
# ============================================================================


def extract_date_from_metadata(metadata: Dict[str, Any]) -> Optional[str]:
    """
    Extract date from metadata fields in priority order.

    Args:
        metadata: Document metadata dictionary

    Returns:
        Date string if found, None otherwise
    """
    for date_field in METADATA_DATE_FIELDS:
        date_val = metadata.get(date_field)

        if date_val and date_val != "publication date unknown":
            parsed = parse_date_string(str(date_val))
            if is_valid_date(parsed):
                logging.debug(
                    "Found date in metadata[%s]: %s", date_field, date_val)
                return str(date_val)

    return None


# ============================================================================
# DATE EXTRACTION FROM TEXT
# ============================================================================


def extract_dates_from_text(text: str) -> List[Tuple[datetime, str, str, int]]:
    """
    Extract all dates from text using pre-compiled patterns.

    Args:
        text: Text content to search

    Returns:
        List of (parsed_date, date_string, pattern_type, priority)
    """
    found_dates = []

    for pattern, pattern_type, priority in DATE_PATTERNS:
        matches = pattern.findall(text)

        for match in matches:
            # Clean up the match
            match = re.sub(r'\s+', ' ', match.strip())

            parsed = parse_date_string(match)

            if is_valid_date(parsed):
                found_dates.append((parsed, match, pattern_type, priority))
                logging.debug(
                    "Found date in text (%s, priority=%d): %s",
                    pattern_type, priority, match
                )

    return found_dates


# ============================================================================
# COMPREHENSIVE DATE SEARCH
# ============================================================================


def search_source_for_latest_date(
    idx,
    key_value: str,
    namespace: str = DEFAULT_NAMESPACE
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Search for all chunks/documents with the same key and find the latest date.

    Args:
        idx: Pinecone index object
        key_value: Document key to search for
        namespace: Pinecone namespace

    Returns:
        (latest_date_str, matching_documents)
    """
    try:
        logging.debug(
            "Searching for dates - Index: %s, Namespace: %s, Key: %s",
            getattr(idx, '_index_name', 'unknown'),
            namespace,
            key_value
        )

        matching_docs = _fetch_document_chunks(idx, key_value, namespace)

        if not matching_docs:
            logging.warning("No chunks found for key='%s'", key_value)
            return None, []

        # Extract dates from all chunks
        all_dates = []
        chunks_with_dates = set()

        for doc in matching_docs:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            doc_id = doc.get("id", "")
            chunk_num = doc.get("chunk", "?")

            # Check metadata first (highest reliability)
            metadata_date = extract_date_from_metadata(metadata)
            if metadata_date:
                parsed = parse_date_string(metadata_date)
                if is_valid_date(parsed):
                    all_dates.append(
                        (parsed, metadata_date, doc_id, "metadata", 20))
                    chunks_with_dates.add(chunk_num)

            # Then search text
            text_dates = extract_dates_from_text(text)
            for parsed, date_str, pattern_type, priority in text_dates:
                all_dates.append(
                    (parsed, date_str, doc_id, pattern_type, priority))
                chunks_with_dates.add(chunk_num)

        # Log summary
        logging.info(
            "Date search for '%s': %d chunks, %d with dates, %d total dates",
            key_value, len(matching_docs), len(
                chunks_with_dates), len(all_dates)
        )

        if not all_dates:
            logging.warning("No valid dates found for key='%s'", key_value)
            return None, matching_docs

        # Sort by priority (desc), then by date (desc)
        all_dates.sort(key=lambda x: (-x[4], -x[0].timestamp()))

        # Get the best date
        latest_date_obj, latest_date_str, doc_id, source_type, priority = all_dates[0]

        logging.info(
            "Selected date: '%s' (source: %s, priority: %d, from: %s)",
            latest_date_str, source_type, priority, doc_id
        )

        # Log alternative candidates for transparency
        if len(all_dates) > 1:
            alternatives = all_dates[1:4]
            logging.debug("Alternative dates: %s", [
                          d[1] for d in alternatives])

        return latest_date_str, matching_docs

    except Exception as e:  # pylint: disable=broad-except
        logging.error(
            "Error searching dates for key='%s': %s",
            key_value, e, exc_info=True
        )
        return None, []


def _fetch_document_chunks(idx, key_value: str, namespace: str) -> List[Dict[str, Any]]:
    """
    Fetch all chunks for a document using metadata filtering or semantic search.

    Args:
        idx: Pinecone index
        key_value: Document key
        namespace: Namespace to search

    Returns:
        List of matching document chunks
    """
    # Strategy 1: Metadata filtering (preferred)
    try:
        raw = idx.query(
            vector=DUMMY_VECTOR,
            filter={"key": {"$eq": key_value}},
            top_k=100,
            namespace=namespace,
            include_metadata=True
        )
        matching_docs = normalise_matches(raw)

        if matching_docs:
            logging.debug(
                "Found %d chunks using metadata filter for key='%s'",
                len(matching_docs), key_value
            )
            return matching_docs

    except Exception as e:  # pylint: disable=broad-except
        logging.debug("Metadata filter failed: %s", e)

    # Strategy 2: Semantic search fallback
    logging.debug("Falling back to semantic search...")

    from pinecone_utils import try_inference_search, vector_query
    from config import DEFAULT_EMBED_MODEL

    try:
        raw = try_inference_search(
            idx, namespace, key_value, k=50, model_name=None)
    except Exception:  # pylint: disable=broad-except
        raw = vector_query(idx, namespace, key_value, 50, DEFAULT_EMBED_MODEL)

    results = normalise_matches(raw)

    # Filter to exact key matches
    matching_docs = [r for r in results if r.get("key") == key_value]

    logging.debug(
        "Found %d chunks via semantic search for key='%s'",
        len(matching_docs), key_value
    )

    return matching_docs


# ============================================================================
# QUICK DATE EXTRACTION
# ============================================================================


def extract_date_from_single_result(result: Dict[str, Any]) -> Optional[str]:
    """
    Extract date from a single search result without additional searches.
    Quick fallback when full document search is not needed.

    Args:
        result: Search result dictionary

    Returns:
        Date string if found, None otherwise
    """
    try:
        metadata = result.get("metadata", {})
        text = result.get("text", "")

        # Check metadata first
        metadata_date = extract_date_from_metadata(metadata)
        if metadata_date:
            return metadata_date

        # Quick text search using pre-compiled patterns
        for pattern in QUICK_DATE_PATTERNS:
            match = pattern.search(text)
            if match:
                date_str = match.group(1)
                parsed = parse_date_string(date_str)
                if is_valid_date(parsed):
                    return date_str

        return None

    except Exception as e:  # pylint: disable=broad-except
        logging.warning("Error extracting date from single result: %s", e)
        return None
