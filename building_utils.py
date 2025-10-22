#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Building name extraction and lookup utilities with dynamic Pinecone index support.
Optimised version with pre-compiled patterns, cached lookups, and alias support.
"""

import re
import json
from typing import Optional, List, Dict, Any
from difflib import get_close_matches
from functools import lru_cache
import logging

# ============================================================================
# BUILDING NAME CACHE (populated from Pinecone at startup)
# ============================================================================

_BUILDING_NAMES_CACHE: Dict[str, str] = {}  # normalized -> canonical
_BUILDING_ALIASES_CACHE: Dict[str, str] = {}  # alias -> canonical
_CACHE_POPULATED = False


def populate_building_cache_from_index(idx, namespace: str = "__default__"):
    """
    Populate building name cache from Pinecone index metadata.
    Call this once at application startup.

    Args:
        idx: Pinecone index object
        namespace: Namespace to query
    """
    global _BUILDING_NAMES_CACHE, _BUILDING_ALIASES_CACHE, _CACHE_POPULATED

    if _CACHE_POPULATED:
        logging.info("Building cache already populated, skipping")
        return

    logging.info("Populating building cache from Pinecone index...")

    try:
        # Query for Planon data records
        dummy_vector = [0.0] * 1536  # Adjust dimension as needed
        results = idx.query(
            vector=dummy_vector,
            filter={"document_type": {"$eq": "planon_data"}},
            top_k=1000,  # Adjust based on number of buildings
            namespace=namespace,
            include_metadata=True
        )

        canonical_names = set()

        for match in results.get("matches", []):
            metadata = match.get("metadata", {})

            # Get canonical name
            canonical = metadata.get("canonical_building_name") or \
                metadata.get("building_name")

            if not canonical:
                continue

            canonical_names.add(canonical)

            # Map normalized canonical name
            normalized = canonical.lower().strip()
            _BUILDING_NAMES_CACHE[normalized] = canonical

            # Map aliases
            aliases = metadata.get("building_aliases", [])
            if isinstance(aliases, list):
                for alias in aliases:
                    if alias:
                        alias_norm = str(alias).lower().strip()
                        _BUILDING_ALIASES_CACHE[alias_norm] = canonical
            elif isinstance(aliases, str):
                # Handle case where aliases might be stored as string
                try:
                    alias_list = json.loads(aliases)  # This now works
                    for alias in alias_list:
                        if alias:
                            alias_norm = str(alias).lower().strip()
                            _BUILDING_ALIASES_CACHE[alias_norm] = canonical
                except (json.JSONDecodeError, TypeError):
                    # If not JSON, treat as single alias
                    alias_norm = aliases.lower().strip()
                    _BUILDING_ALIASES_CACHE[alias_norm] = canonical

        _CACHE_POPULATED = True

        logging.info(
            "Building cache populated: %d canonical names, %d aliases",
            len(canonical_names),
            len(_BUILDING_ALIASES_CACHE)
        )

    except Exception as e:  # pylint: disable=broad-except
        logging.error("Failed to populate building cache: %s",
                      e, exc_info=True)
        _CACHE_POPULATED = False


def get_building_names_from_cache() -> List[str]:
    """Get list of canonical building names from cache."""
    return list(set(_BUILDING_NAMES_CACHE.values()))


def clear_building_cache():
    """Clear building name cache (useful for testing or reloading)."""
    global _BUILDING_NAMES_CACHE, _BUILDING_ALIASES_CACHE, _CACHE_POPULATED
    _BUILDING_NAMES_CACHE = {}
    _BUILDING_ALIASES_CACHE = {}
    _CACHE_POPULATED = False
    logging.info("Building cache cleared")


def get_cache_status() -> Dict[str, Any]:
    """Get building cache status for debugging."""
    return {
        "populated": _CACHE_POPULATED,
        "canonical_names": len(set(_BUILDING_NAMES_CACHE.values())),
        "total_mappings": len(_BUILDING_NAMES_CACHE),
        "aliases": len(_BUILDING_ALIASES_CACHE),
    }


# ============================================================================
# MODULE-LEVEL CONSTANTS (compiled once at import time)
# ============================================================================

# Pre-compile regex patterns for efficiency
# Pre-compile regex patterns for efficiency
BUILDING_PATTERNS = [
    # Pattern for "at/in/for/about [Building]" - most reliable
    re.compile(
        r'(?:at|in|for|about|from)\s+([A-Z0-9][A-Za-z0-9\s\-]+?)(?:\s+(?:building|bms|controls|have|has)|\?|$|\.)',
        re.IGNORECASE
    ),
    # Pattern for addresses with numbers: "1-9 Old Park Hill"
    re.compile(
        r'\b(\d+(?:-\d+)?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b',
        re.IGNORECASE
    ),
    # Pattern for "building/property/site [Name]"
    re.compile(
        r'(?:building|property|site)\s+([A-Z0-9][A-Za-z0-9\s]+?)(?:\s+(?:building|bms|controls|system)|\?|$)',
        re.IGNORECASE
    ),
    # Pattern for proper nouns (two capitalized words) - BUT not at sentence start
    re.compile(
        r'(?<!^)(?<!\.\s)\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
        re.IGNORECASE
    ),
]

# Pre-compiled normalization patterns
NORMALIZATION_PATTERNS = [
    re.compile(r'\s+BMS.*$', re.IGNORECASE),
    re.compile(r'\s+Controls.*$', re.IGNORECASE),
    re.compile(r'\s+Project.*$', re.IGNORECASE),
    re.compile(r'\s+Manual.*$', re.IGNORECASE),
    re.compile(r'\s+-\s+.*$', re.IGNORECASE),
    re.compile(r'\s+FRA.*$', re.IGNORECASE),
]

# Pre-compile article removal pattern
ARTICLE_PATTERN = re.compile(r'\s+(the|a|an)$', re.IGNORECASE)

# Minimum building name length for filtering
MIN_BUILDING_NAME_LENGTH = 3  # Reduced from 4 to catch "SHB" and similar

# Stop words to skip (question/action words that aren't buildings)
QUESTION_WORDS = frozenset({
    'tell', 'does', 'do', 'did', 'can', 'could', 'would', 'should',
    'will', 'what', 'when', 'where', 'who', 'why', 'how', 'which',
    'is', 'are', 'was', 'were', 'has', 'have', 'had', 'show', 'list',
    'find', 'get', 'give', 'make', 'help', 'please', 'provide', 'need',
    'looking', 'searching', 'information', 'info', 'about', 'on', 'for',
    'at', 'in', 'of', 'the', 'a', 'an', 'hello', 'hi', 'me'
})


# ============================================================================
# CORE FUNCTIONS
# ============================================================================


def extract_building_from_query(
    query: str,
    known_buildings: Optional[List[str]] = None,
    use_cache: bool = True
) -> Optional[str]:
    """
    Extract building name from user query using multiple strategies.
    Uses cached building data from Pinecone if available.

    Args:
        query: User query string
        known_buildings: Optional list of known building names for fuzzy matching
        use_cache: Whether to use cached building names (default True)

    Returns:
        Canonical building name or None if not found
    """
    if not query:
        return None

    query_lower = query.lower()

    # Strategy 1: Check cached canonical names (exact substring match)
    if use_cache and _CACHE_POPULATED:
        for normalized, canonical in _BUILDING_NAMES_CACHE.items():
            # Check if building name appears as whole words in query
            pattern = r'\b' + re.escape(normalized) + r'\b'
            if re.search(pattern, query_lower):
                logging.info(
                    "Found building in query (canonical): '%s'", canonical)
                return canonical

    # Strategy 2: Check aliases
    if use_cache and _CACHE_POPULATED:
        for alias_norm, canonical in _BUILDING_ALIASES_CACHE.items():
            # Check for exact alias match
            pattern = r'\b' + re.escape(alias_norm) + r'\b'
            if re.search(pattern, query_lower):
                logging.info("Found building via alias '%s' -> '%s'",
                             alias_norm, canonical)
                return canonical

    # Strategy 3: Pattern-based extraction
    building_candidate = _extract_with_patterns(query)
    if building_candidate:
        # Try to match against cache
        if use_cache and _CACHE_POPULATED:
            matched = find_closest_building_name(
                building_candidate,
                get_building_names_from_cache()
            )
            if matched:
                logging.info("Pattern extracted and matched: '%s' -> '%s'",
                             building_candidate, matched)
                return matched

        # Try against provided known_buildings
        if known_buildings:
            matched = find_closest_building_name(
                building_candidate, known_buildings)
            if matched:
                logging.info("Pattern extracted and matched (known): '%s' -> '%s'",
                             building_candidate, matched)
                return matched

        logging.info("Pattern extracted (no match): '%s'", building_candidate)
        return building_candidate

    # Strategy 4: Search in known buildings (only if provided)
    if known_buildings:
        result = _find_in_known_buildings(query_lower, known_buildings)
        if result:
            logging.info("Found known building in query: '%s'", result)
            return result

    return None


def _extract_with_patterns(query: str) -> Optional[str]:
    """
    Extract building name using pre-compiled regex patterns.
    Helper function to keep main function clean.
    """
    for pattern in BUILDING_PATTERNS:
        match = pattern.search(query)
        if match:
            building_candidate = match.group(1).strip()

            # Clean up the candidate
            building_candidate = ARTICLE_PATTERN.sub('', building_candidate)

            if not building_candidate:
                continue

            # Skip if ANY word is a question word (not just the first)
            words = building_candidate.lower().split()
            if any(word in QUESTION_WORDS for word in words):
                logging.debug(
                    "Skipping candidate with question words: '%s'", building_candidate)
                continue

            # Skip if it's too generic (only common words)
            if all(word in QUESTION_WORDS for word in words):
                continue

            # Validate: reasonable building name length (1-4 words)
            word_count = len(words)
            if 1 <= word_count <= 4 and len(building_candidate) >= MIN_BUILDING_NAME_LENGTH:
                return building_candidate

    return None


def _find_in_known_buildings(
    query_lower: str,
    known_buildings: List[str]
) -> Optional[str]:
    """
    Find building name in known buildings list.
    Optimized with length filtering and early exit.
    """
    # Filter and search in one pass
    for building in known_buildings:
        if len(building) >= MIN_BUILDING_NAME_LENGTH and building.lower() in query_lower:
            return building

    return None


@lru_cache(maxsize=1024)
def normalise_building_name(building_name: str) -> str:
    """
    Normalise a building name by removing common suffixes.
    Cached for repeated calls with same input.

    E.g., "Senate House BMS Controls" -> "Senate House"

    Args:
        building_name: Original building name

    Returns:
        Normalised building name
    """
    if not building_name:
        return building_name

    normalised = building_name
    for pattern in NORMALIZATION_PATTERNS:
        normalised = pattern.sub('', normalised)

    return normalised.strip()


def find_closest_building_name(
    extracted_name: str,
    known_buildings: Optional[List[str]] = None
) -> Optional[str]:
    """
    Find the closest matching building name using cached data and fuzzy matching.

    Args:
        extracted_name: Building name extracted from query
        known_buildings: Optional list of known building names (uses cache if None)

    Returns:
        Best matching canonical building name or None
    """
    if not extracted_name:
        return None

    # Use cache if no buildings provided
    if known_buildings is None:
        if _CACHE_POPULATED:
            known_buildings = get_building_names_from_cache()
        else:
            return None

    if not known_buildings:
        return None

    # Cache the lowercase version for multiple comparisons
    extracted_lower = extracted_name.lower()

    # Strategy 1: Check aliases first (if cache populated)
    if _CACHE_POPULATED:
        canonical = _BUILDING_ALIASES_CACHE.get(extracted_lower)
        if canonical:
            logging.info("Alias match: '%s' -> '%s'",
                         extracted_name, canonical)
            return canonical

        # Check canonical names
        canonical = _BUILDING_NAMES_CACHE.get(extracted_lower)
        if canonical:
            logging.info("Exact match: '%s'", canonical)
            return canonical

    # Strategy 2: Exact match (case-insensitive)
    for building in known_buildings:
        if building.lower() == extracted_lower:
            return building

    # Strategy 3: Substring match (extracted name in building name)
    for building in known_buildings:
        if extracted_lower in building.lower():
            return building

    # Strategy 4: Reverse substring match (building name in extracted name)
    for building in known_buildings:
        if building.lower() in extracted_lower:
            return building

    # Strategy 5: Fuzzy match using difflib (80% similarity)
    matches = get_close_matches(
        extracted_name, known_buildings, n=1, cutoff=0.80
    )
    if matches:
        return matches[0]

    return None


def group_results_by_building(
    results: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group search results by normalised building name.
    This ensures "Senate House" and "Senate House BMS Controls" are grouped together.

    Args:
        results: List of search results with building_name field

    Returns:
        Dictionary mapping normalised building names to result lists
    """
    grouped = {}

    for result in results:
        building_name = result.get('building_name', 'Unknown')

        # Normalise the building name for grouping
        normalised_name = normalise_building_name(building_name)

        # If normalization made it empty, use original
        if not normalised_name:
            normalised_name = building_name

        if normalised_name not in grouped:
            grouped[normalised_name] = []

        # Add normalized building name for reference (modify in place)
        # No need for copy since we're just adding metadata
        if '_normalised_building' not in result:
            result['_normalised_building'] = normalised_name

        grouped[normalised_name].append(result)

    return grouped


def get_building_context_summary(
    building_results: Dict[str, List[Dict[str, Any]]]
) -> str:
    """
    Create a summary of buildings found in search results.

    Args:
        building_results: Results grouped by building name

    Returns:
        Formatted summary string
    """
    if not building_results:
        return ""

    summary_parts = []
    for building, results in building_results.items():
        # Count document types efficiently
        doc_types = {}
        for r in results:
            doc_type = r.get('document_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

        # Format document type counts
        type_str = ', '.join(
            f"{count} {dtype}" for dtype, count in sorted(doc_types.items())
        )
        summary_parts.append(
            f"- {building}: {len(results)} results ({type_str})"
        )

    return "Buildings found:\n" + "\n".join(summary_parts)


def prioritise_building_results(
    results: List[Dict[str, Any]],
    target_building: str
) -> List[Dict[str, Any]]:
    """
    Reorder results to prioritise a specific building.
    Uses flexible matching to catch name variations.
    Optimized with cached normalizations and single-pass sorting.

    Args:
        results: List of search results
        target_building: Building name to prioritise

    Returns:
        Reordered list with target building results first
    """
    if not target_building or not results:
        return results

    # Pre-compute target variations once
    target_normalised = normalise_building_name(target_building).lower()
    target_lower = target_building.lower()

    # Single-pass separation with cached normalizations
    priority_results = []
    other_results = []

    for result in results:
        building_name = result.get('building_name', '')

        # Use cached normalization if available
        if '_normalised_building' in result:
            normalised = result['_normalised_building'].lower()
        else:
            normalised = normalise_building_name(building_name).lower()

        building_lower = building_name.lower()

        # Check if this result matches the target building
        is_match = (
            target_normalised in normalised or
            normalised in target_normalised or
            target_lower in building_lower or
            building_lower in target_lower
        )

        if is_match:
            priority_results.append(result)
        else:
            other_results.append(result)

    logging.info(
        "Prioritised %d results for '%s', %d other results",
        len(priority_results),
        target_building,
        len(other_results)
    )

    # Return combined list with priority results first
    return priority_results + other_results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_building_variations(building_name: str) -> List[str]:
    """
    Generate common variations of a building name for matching.

    Args:
        building_name: Original building name

    Returns:
        List of name variations
    """
    variations = [building_name]

    # Add normalized version
    normalized = normalise_building_name(building_name)
    if normalized != building_name:
        variations.append(normalized)

    # Add without common words
    for word in ['Building', 'House', 'Hall', 'Complex']:
        if word in building_name:
            variation = building_name.replace(word, '').strip()
            if variation and variation not in variations:
                variations.append(variation)

    return variations


def clear_caches():
    """Clear all LRU caches (useful for testing or memory management)."""
    normalise_building_name.cache_clear()
    clear_building_cache()
    logging.info("Cleared building_utils caches")


def get_cache_info() -> Dict[str, Any]:
    """Get information about cache usage for monitoring."""
    return {
        'normalise_building_name': normalise_building_name.cache_info()._asdict(),
        'building_cache': get_cache_status()
    }
