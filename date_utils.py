#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date parsing and publication date search utilities.
"""

import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from pinecone_utils import normalize_matches
from config import DEFAULT_NAMESPACE


def parse_date_string(date_str: str) -> datetime:
    """Parse various date formats and return a datetime object."""
    if not date_str or date_str == "publication date unknown":
        return datetime.min

    # Common date formats to try (including dot format and UK format)
    formats = [
        "%d %B %Y",  # 03 November 2021
        "%d.%m.%Y",  # 12.06.2025 (dot format)
        "%Y.%m.%d",  # 2025.06.12 (dot format)
        "%Y-%m-%d",  # 2025-07-28
        "%B %d, %Y",  # July 28, 2025
        "%d/%m/%Y",  # 28/07/2025
        "%m/%d/%Y",  # 07/28/2025
        "%Y",  # 2025
        "%B %Y",  # July 2025
        "%d-%m-%Y",  # 28-07-2025
        "%Y/%m/%d",  # 2025/07/28
        "%d %b %Y",  # 19 Mar 2020 (abbreviated month)
        "%d-%b-%Y",  # 19-Mar-2020
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue

    # If no format matches, return the minimum date
    return datetime.min


def format_display_date(date_obj: datetime) -> str:
    """Format datetime object for display."""
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


def search_source_for_latest_date(idx, key_value: str, namespace: str = DEFAULT_NAMESPACE) -> Tuple[
        Optional[str], List[Dict[str, Any]]]:
    """
    Search for all chunks/documents with the same key and determine the latest publication/review date
    """
    try:
        logging.info(f"=" * 60)
        logging.info(f"SEARCHING FOR DATE:")
        logging.info(
            f"  Index: {idx._index_name if hasattr(idx, '_index_name') else 'unknown'}")
        logging.info(f"  Namespace: {namespace}")
        logging.info(f"  Key: {key_value}")
        logging.info(f"=" * 60)

        # Strategy 1: Try metadata filtering to get all chunks from this document
        matching_docs = []
        try:
            # Use metadata filter if supported
            raw = idx.query(
                vector=[0.0] * 1536,  # Dummy vector for metadata-only query
                filter={"key": {"$eq": key_value}},
                top_k=100,  # Get many chunks to cover the whole document
                namespace=namespace,
                include_metadata=True
            )
            matching_docs = normalize_matches(raw)
            logging.info(
                f"✓ Found {len(matching_docs)} chunks using metadata filter for key='{key_value}'")
        except Exception as e:
            logging.warning(f"Metadata filter not supported or failed: {e}")
            matching_docs = []

        # Strategy 2: Fallback to semantic search if metadata filtering didn't work
        if not matching_docs:
            logging.info(
                "Falling back to semantic search with key filtering...")
            from pinecone_utils import try_inference_search, vector_query
            from config import DEFAULT_EMBED_MODEL

            # Search with the document key/name
            source_query = f"{key_value}"

            try:
                raw = try_inference_search(
                    idx, namespace, source_query, k=50, model_name=None)
            except Exception:
                raw = vector_query(
                    idx, namespace, source_query, 50, DEFAULT_EMBED_MODEL)

            results = normalize_matches(raw)

            # Filter to exact key matches only
            matching_docs = [r for r in results if r.get("key") == key_value]
            logging.info(
                f"✓ Found {len(matching_docs)} chunks matching key='{key_value}' after filtering")

        if not matching_docs:
            logging.warning(f"✗ No chunks found for key='{key_value}'")
            return None, []

        # Extract all possible dates from ALL chunks of this document
        all_dates = []

        # Enhanced date patterns with priority weighting
        # Higher priority = more reliable date source
        date_patterns = [
            # Most reliable: explicitly labeled dates in common document formats
            (r'(?:Last\s+Updated|Last\s+Revised|Date\s+Updated|Date\s+Revised)[:\s]+([0-3]?[0-9][\s/.-][A-Za-z]+[\s/.-][0-9]{4})', 'labeled_updated', 15),
            (r'(?:Last\s+Updated|Last\s+Revised|Date\s+Updated|Date\s+Revised)[:\s]+([0-3]?[0-9][\s/.-][0-3]?[0-9][\s/.-][0-9]{4})', 'labeled_updated_numeric', 15),
            (r'(?:Updated|Revised|Review\s+Date|Publication\s+Date)[:\s]+([0-3]?[0-9][\s/.-][A-Za-z]+[\s/.-][0-9]{4})', 'labeled_general', 12),
            (r'(?:Updated|Revised)[:\s]+([0-3]?[0-9][\s/.-][0-3]?[0-9][\s/.-][0-9]{4})', 'labeled_general_numeric', 12),

            # Document header patterns (common in technical documents)
            (r'(?:Document\s+Date|Issue\s+Date|Effective\s+Date)[:\s]+([0-3]?[0-9][\s/.-][A-Za-z]+[\s/.-][0-9]{4})', 'doc_header', 14),
            (r'(?:Version|Rev\.?\s+|Revision)[:\s]*([0-9]{1,2}[\s/.-][A-Z][a-z]{2}[\s/.-][0-9]{4})', 'version_date', 13),

            # Dot format patterns (common in UK/European documents)
            (r'\b([0-3]?[0-9]\.[0-1]?[0-9]\.[0-9]{4})\b', 'dot_dmy', 8),
            (r'\b([0-9]{4}\.[0-1]?[0-9]\.[0-3]?[0-9])\b', 'dot_ymd', 8),

            # Standard text dates
            (r'\b([0-3]?[0-9][\s/][A-Za-z]{3,9}[\s/][0-9]{4})\b',
             'standard_text', 7),
            (r'\b([0-3]?[0-9][\s][A-Z][a-z]{2}[\s][0-9]{4})\b',
             'abbreviated_text', 7),

            # Numeric date formats
            (r'\b([0-3]?[0-9][/-][0-1]?[0-9][/-][0-9]{4})\b',
             'standard_numeric', 6),
            (r'\b([0-9]{4}[/-][0-1]?[0-9][/-][0-3]?[0-9])\b', 'iso_date', 6),

            # Copyright/revision patterns (lower priority as they might be template dates)
            (r'(?:©|Copyright)[:\s]*([0-9]{4})', 'copyright_year', 3),
            (r'(?:Rev\.?\s*|Version\s+)[:\s]*([0-9]{4})', 'version_year', 3),
        ]

        # Track which chunks have dates for logging
        chunks_with_dates = set()

        for doc in matching_docs:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            doc_id = doc.get("id", "")
            chunk_num = doc.get("chunk", "?")

            # First, check metadata fields (highest reliability)
            # Skip publication_date as it's known to be unreliable
            for date_field in ["last_modified", "review_date", "updated", "revised", "date", "document_date"]:
                if date_field in metadata:
                    date_val = metadata[date_field]
                    if date_val and date_val != "publication date unknown":
                        parsed = parse_date_string(str(date_val))
                        if parsed != datetime.min:
                            all_dates.append(
                                (parsed, str(date_val), doc_id, f"metadata:{date_field}", 20))
                            chunks_with_dates.add(chunk_num)
                            logging.info(
                                f"  ✓ Chunk {chunk_num}: Found date in metadata[{date_field}]: {date_val}")

            # Then search through text with prioritized patterns
            for pattern, pattern_type, priority in date_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Clean up the match (remove extra spaces)
                    match = re.sub(r'\s+', ' ', match.strip())

                    parsed = parse_date_string(match)
                    if parsed != datetime.min:
                        # Validate date is reasonable
                        current_date = datetime.now()

                        # Date should be:
                        # 1. Not in the future
                        # 2. Not too old (within 30 years is reasonable for technical docs)
                        # 3. Not suspiciously old (before 1990 is unlikely for BMS docs)
                        if parsed <= current_date and (current_date - parsed).days < 30 * 365 and parsed.year >= 1990:
                            all_dates.append(
                                (parsed, match, doc_id, pattern_type, priority))
                            chunks_with_dates.add(chunk_num)
                            logging.info(
                                f"  ✓ Chunk {chunk_num}: Found date in text ({pattern_type}, priority={priority}): {match}")

        # Summary logging
        logging.info(f"📊 Date search summary for '{key_value}':")
        logging.info(f"   - Total chunks searched: {len(matching_docs)}")
        logging.info(f"   - Chunks with dates found: {len(chunks_with_dates)}")
        logging.info(f"   - Total dates extracted: {len(all_dates)}")

        if not all_dates:
            logging.warning(
                f"✗ No valid dates found in any of the {len(matching_docs)} chunks for key='{key_value}'")
            return None, matching_docs

        # Sort by priority first (highest priority first), then by date (newest first)
        all_dates.sort(key=lambda x: (-x[4], -x[0].timestamp()))

        # Get the highest priority, most recent date
        latest_date_obj, latest_date_str, doc_id, source_type, priority = all_dates[0]

        logging.info(f"✓ SELECTED DATE: '{latest_date_str}'")
        logging.info(f"   - Source type: {source_type}")
        logging.info(f"   - Priority score: {priority}")
        logging.info(f"   - From chunk ID: {doc_id}")
        logging.info(f"   - Parsed as: {format_display_date(latest_date_obj)}")

        # Log top 3 candidates for transparency
        if len(all_dates) > 1:
            logging.info(f"   - Other candidates found:")
            for i, (date_obj, date_str, doc_id, src_type, prio) in enumerate(all_dates[1:4], 1):
                logging.info(
                    f"     {i}. {date_str} (source: {src_type}, priority: {prio})")

        return latest_date_str, matching_docs

    except Exception as e:
        logging.error(
            f"✗ Error searching for source dates for key='{key_value}': {e}", exc_info=True)
        return None, []


def extract_date_from_single_result(result: Dict[str, Any]) -> Optional[str]:
    """
    Extract date from a single search result without doing additional searches.
    Used as a quick fallback when full document search is not needed.

    Returns:
        Date string if found, None otherwise
    """
    try:
        metadata = result.get("metadata", {})
        text = result.get("text", "")

        # Check metadata first
        for date_field in ["last_modified", "review_date", "updated", "revised", "date"]:
            if date_field in metadata:
                date_val = metadata[date_field]
                if date_val and date_val != "publication date unknown":
                    parsed = parse_date_string(str(date_val))
                    if parsed != datetime.min:
                        return str(date_val)

        # Quick text search for labeled dates
        patterns = [
            r'(?:Last\s+Updated|Last\s+Revised|Updated|Revised)[:\s]+([0-3]?[0-9][\s/.-][A-Za-z]+[\s/.-][0-9]{4})',
            r'(?:Last\s+Updated|Last\s+Revised|Updated|Revised)[:\s]+([0-3]?[0-9][\s/.-][0-3]?[0-9][\s/.-][0-9]{4})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                parsed = parse_date_string(date_str)
                if parsed != datetime.min:
                    return date_str

        return None

    except Exception as e:
        logging.warning(f"Error extracting date from single result: {e}")
        return None
