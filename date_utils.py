#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date parsing and publication date search utilities.
"""

import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from pinecone_utils import try_inference_search, vector_query, normalize_matches
from config import DEFAULT_NAMESPACE, DEFAULT_EMBED_MODEL


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
    Search for all documents from the same key and determine the latest publication/review date.

    Returns:
        Tuple of (latest_date_string, all_matching_documents)
    """
    try:
        # Query for documents with the same key
        source_query = f"key:{key_value} publication date review date updated revised"

        # Try to get many results to find all documents from this source
        try:
            raw = try_inference_search(
                idx, namespace, source_query, k=20, model_name=None)
        except:
            # Fallback to vector search
            raw = vector_query(idx, namespace, source_query,
                               20, DEFAULT_EMBED_MODEL)

        results = normalize_matches(raw)

        # Filter results to only those from the same key
        matching_docs = [r for r in results if r.get("key") == key_value]

        if not matching_docs:
            # If no exact matches, try without filtering
            matching_docs = results[:10]  # Take top 10 results

        # Extract all possible dates from matching documents
        all_dates = []
        date_patterns = [
            # Dot format patterns
            (r'\b(\d{1,2}\.\d{1,2}\.\d{4})\b', 'dot_dmy'),
            (r'\b(\d{4}\.\d{1,2}\.\d{1,2})\b', 'dot_ymd'),
            # Context-aware patterns
            (r'\b(?:published|updated|revised|reviewed)[\s:]+(\d{1,2}[/.]\d{1,2}[/.]\d{4})\b', 'full'),
            (r'\b(?:published|updated|revised|reviewed)[\s:]+(\d{4}[/.-]\d{1,2}[/.-]\d{1,2})\b', 'iso'),
            (r'\b(?:published|updated|revised|reviewed)[\s:]+(?:in\s+)?(\d{4})\b', 'year'),
            (r'\b(?:Publication Date|Review Date|Last Updated|Last Revised)[\s:]+([^,\n]+)', 'labeled'),
            (r'\b(?:©|Copyright)\s+(\d{4})\b', 'copyright'),
            # Generic patterns
            (r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b', 'generic_full'),
            (r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b', 'generic_iso'),
        ]

        for doc in matching_docs:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})

            # Check metadata fields EXCEPT publication_date (as it's misleading)
            for date_field in ["review_date", "updated", "revised", "date", "last_modified"]:
                if date_field in metadata:
                    date_val = metadata[date_field]
                    if date_val and date_val != "publication date unknown":
                        parsed = parse_date_string(str(date_val))
                        if parsed != datetime.min:
                            all_dates.append(
                                (parsed, str(date_val), doc.get("id")))

            # Check text content with context-aware patterns
            for pattern, pattern_type in date_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    parsed = parse_date_string(match)
                    if parsed != datetime.min:
                        # Validate date is reasonable (not in future, not too old)
                        current_date = datetime.now()
                        if parsed <= current_date and (current_date - parsed).days < 20 * 365:
                            all_dates.append((parsed, match, doc.get("id")))

        # Find the latest date
        if all_dates:
            all_dates.sort(key=lambda x: x[0], reverse=True)
            latest_date_obj, latest_date_str, doc_id = all_dates[0]
            return latest_date_str, matching_docs

        return None, matching_docs

    except Exception as e:
        logging.error(f"Error searching for source dates: {e}")
        return None, []
