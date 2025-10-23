#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Building name extraction and lookup utilities with dynamic Pinecone index support.
IMPROVED VERSION: Enhanced fuzzy matching against multiple metadata fields.

Key improvements:
- Fuzzy matching (80% threshold) against multiple metadata fields
- Support for Property names, UsrFRACondensedPropertyName, building_name, canonical_building_name
- Enhanced metadata search strategy for better building detection
- Better handling of variations and aliases
"""

import re
import json
from typing import Optional, List, Dict, Any, Set
from difflib import get_close_matches, SequenceMatcher
from functools import lru_cache
import logging

# ============================================================================
# BUILDING NAME CACHE (populated from Pinecone at startup)
# ============================================================================

_BUILDING_NAMES_CACHE: Dict[str, str] = {}  # normalised -> canonical
_BUILDING_ALIASES_CACHE: Dict[str, str] = {}  # alias -> canonical
# canonical -> set of all known variations from metadata
_METADATA_FIELDS_CACHE: Dict[str, Set[str]] = {}
_CACHE_POPULATED = False
_INDEXES_WITH_BUILDINGS: List[str] = []

# Metadata fields to search for building names (in priority order)
BUILDING_METADATA_FIELDS = [
    'canonical_building_name',
    'building_name',
    'Property names',
    'UsrFRACondensedPropertyName',
    'building_aliases'
]

# Fuzzy match threshold (80% similarity)
FUZZY_MATCH_THRESHOLD = 0.80

# ============================================================================
# BUILDING PATTERNS (for query extraction)
# ============================================================================

# Common building name patterns
BUILDING_PATTERNS = [
    # "at <building>" pattern
    re.compile(
        r'\bat\s+([A-Z][A-Za-z\s\-\']+(?:Building|House|Hall|Centre|Center|Complex|Tower)?)', re.IGNORECASE),
    # "in <building>" pattern
    re.compile(
        r'\bin\s+([A-Z][A-Za-z\s\-\']+(?:Building|House|Hall|Centre|Center|Complex|Tower)?)', re.IGNORECASE),
    # "for <building>" pattern
    re.compile(
        r'\bfor\s+([A-Z][A-Za-z\s\-\']+(?:Building|House|Hall|Centre|Center|Complex|Tower)?)', re.IGNORECASE),
    # "<building> building/house" pattern
    re.compile(
        r'\b([A-Z][A-Za-z\s\-\']+)\s+(?:Building|House|Hall|Centre|Center)', re.IGNORECASE),
    # Capitalized words (potential building names)
    re.compile(r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\b'),
]

# Question words to filter out
QUESTION_WORDS = frozenset({
    'what', 'when', 'where', 'which', 'who', 'how', 'why', 'tell', 'show',
    'find', 'search', 'get', 'give', 'list', 'are', 'is', 'do', 'does',
    'can', 'could', 'would', 'should', 'fire', 'risk', 'assessment'
})

# Minimum length for building names
MIN_BUILDING_NAME_LENGTH = 3


# ============================================================================
# CACHE MANAGEMENT
# ============================================================================


def populate_building_cache_from_multiple_indexes(
    index_names: List[str],
    namespace: str = "__default__"
) -> Dict[str, int]:
    """
    Populate building name cache from multiple Pinecone indexes.
    Tries all indexes and aggregates results.

    Args:
        index_names: List of index names to try
        namespace: Namespace to query

    Returns:
        Dictionary mapping index names to number of buildings found
    """
    global _CACHE_POPULATED, _INDEXES_WITH_BUILDINGS

    results = {}
    total_buildings = 0

    # Import here to avoid circular dependency
    from pinecone_utils import open_index

    for idx_name in index_names:
        try:
            logging.info("Trying to populate cache from index '%s'", idx_name)
            idx = open_index(idx_name)
            buildings_found = populate_building_cache_from_index(
                idx, namespace, index_name=idx_name, skip_if_populated=False
            )

            results[idx_name] = buildings_found
            total_buildings += buildings_found

            if buildings_found > 0:
                _INDEXES_WITH_BUILDINGS.append(idx_name)
                logging.info("✅ Index '%s' has building data (%d buildings)",
                             idx_name, buildings_found)
            else:
                logging.info("⚠️  Index '%s' has no building data", idx_name)

        except Exception as e:  # pylint: disable=broad-except
            logging.warning("Failed to check index '%s': %s", idx_name, e)
            results[idx_name] = 0

    if total_buildings > 0:
        _CACHE_POPULATED = True
        logging.info(
            "✅ Building cache initialised from %d/%d indexes: %d total buildings",
            len(_INDEXES_WITH_BUILDINGS),
            len(index_names),
            total_buildings
        )
    else:
        logging.warning(
            "⚠️  No building data found in any of %d indexes",
            len(index_names)
        )
        _CACHE_POPULATED = False

    return results


def populate_building_cache_from_index(
    idx,
    namespace: str = "__default__",
    index_name: Optional[str] = None,
    skip_if_populated: bool = True
) -> int:
    """
    Populate building name cache from Pinecone index metadata.
    IMPROVED: Extracts data from multiple metadata fields.

    Args:
        idx: Pinecone index object
        namespace: Namespace to query
        index_name: Name of the index (optional)
        skip_if_populated: If True, skip if cache already populated

    Returns:
        Number of buildings found in this index
    """
    global _BUILDING_NAMES_CACHE, _BUILDING_ALIASES_CACHE, _METADATA_FIELDS_CACHE, _CACHE_POPULATED

    if skip_if_populated and _CACHE_POPULATED:
        logging.info("Building cache already populated, skipping")
        return len(set(_BUILDING_NAMES_CACHE.values()))

    logging.info("Attempting to populate building cache from index '%s'...",
                 index_name or "unknown")

    try:
        # Import config here to avoid circular imports
        from config import get_index_config

        # Determine the index name
        resolved_index_name: str = index_name or 'operational-docs'

        # Get the correct dimension for this index
        index_config = get_index_config(resolved_index_name)
        dimension = index_config['dimension']
        logging.info("Using dimension %d for index '%s'",
                     dimension, resolved_index_name)

        # Query for Planon data records
        dummy_vector = [0.0] * dimension
        results = idx.query(
            vector=dummy_vector,
            filter={"document_type": {"$eq": "planon_data"}},
            top_k=1000,  # Adjust based on number of buildings
            namespace=namespace,
            include_metadata=True
        )

        matches = results.get("matches", [])
        if not matches:
            logging.info("No planon_data found in index '%s'",
                         resolved_index_name)
            return 0

        canonical_names = set()
        new_names_count = 0
        new_aliases_count = 0
        new_metadata_fields_count = 0

        for match in matches:
            metadata = match.get("metadata", {})

            # Extract canonical name (priority order)
            canonical = None
            for field in ['canonical_building_name', 'building_name']:
                if metadata.get(field):
                    canonical = metadata[field]
                    break

            if not canonical:
                continue

            canonical_names.add(canonical)

            # Initialise metadata fields set for this building
            if canonical not in _METADATA_FIELDS_CACHE:
                _METADATA_FIELDS_CACHE[canonical] = set()

            # Map normalised canonical name
            normalised = canonical.lower().strip()
            if normalised not in _BUILDING_NAMES_CACHE:
                _BUILDING_NAMES_CACHE[normalised] = canonical
                new_names_count += 1

            # Add canonical to metadata fields
            _METADATA_FIELDS_CACHE[canonical].add(canonical)
            _METADATA_FIELDS_CACHE[canonical].add(normalised)

            # Extract and store ALL variations from metadata fields
            for field in BUILDING_METADATA_FIELDS:
                field_value = metadata.get(field)

                if not field_value:
                    continue

                # Handle list of values
                if isinstance(field_value, list):
                    for value in field_value:
                        if value:
                            value_str = str(value).strip()
                            if value_str:
                                value_norm = value_str.lower()
                                _METADATA_FIELDS_CACHE[canonical].add(
                                    value_str)
                                _METADATA_FIELDS_CACHE[canonical].add(
                                    value_norm)

                                # Also add to aliases cache
                                if value_norm not in _BUILDING_ALIASES_CACHE:
                                    _BUILDING_ALIASES_CACHE[value_norm] = canonical
                                    new_aliases_count += 1

                                new_metadata_fields_count += 1

                # Handle string (might be JSON)
                elif isinstance(field_value, str):
                    # Try to parse as JSON first
                    try:
                        value_list = json.loads(field_value)
                        if isinstance(value_list, list):
                            for value in value_list:
                                if value:
                                    value_str = str(value).strip()
                                    if value_str:
                                        value_norm = value_str.lower()
                                        _METADATA_FIELDS_CACHE[canonical].add(
                                            value_str)
                                        _METADATA_FIELDS_CACHE[canonical].add(
                                            value_norm)

                                        if value_norm not in _BUILDING_ALIASES_CACHE:
                                            _BUILDING_ALIASES_CACHE[value_norm] = canonical
                                            new_aliases_count += 1

                                        new_metadata_fields_count += 1
                        else:
                            # Single value from JSON
                            value_str = str(value_list).strip()
                            if value_str:
                                value_norm = value_str.lower()
                                _METADATA_FIELDS_CACHE[canonical].add(
                                    value_str)
                                _METADATA_FIELDS_CACHE[canonical].add(
                                    value_norm)

                                if value_norm not in _BUILDING_ALIASES_CACHE:
                                    _BUILDING_ALIASES_CACHE[value_norm] = canonical
                                    new_aliases_count += 1

                                new_metadata_fields_count += 1
                    except (json.JSONDecodeError, TypeError):
                        # Not JSON, treat as single string value
                        value_str = field_value.strip()
                        if value_str:
                            value_norm = value_str.lower()
                            _METADATA_FIELDS_CACHE[canonical].add(value_str)
                            _METADATA_FIELDS_CACHE[canonical].add(value_norm)

                            if value_norm not in _BUILDING_ALIASES_CACHE:
                                _BUILDING_ALIASES_CACHE[value_norm] = canonical
                                new_aliases_count += 1

                            new_metadata_fields_count += 1

        buildings_found = len(canonical_names)

        if buildings_found > 0:
            _CACHE_POPULATED = True
            logging.info(
                "✅ Index '%s': Found %d buildings (%d canonical names, %d aliases, %d metadata fields)",
                resolved_index_name,
                buildings_found,
                new_names_count,
                new_aliases_count,
                new_metadata_fields_count
            )
        else:
            logging.info(
                "⚠️  Index '%s': No building data found",
                resolved_index_name
            )

        return buildings_found

    except Exception as e:  # pylint: disable=broad-except
        logging.warning(
            "Failed to populate building cache from index '%s': %s",
            index_name or "unknown",
            e
        )
        return 0


def get_indexes_with_buildings() -> List[str]:
    """
    Get list of indexes that have building data.

    Returns:
        List of index names with building data
    """
    return _INDEXES_WITH_BUILDINGS.copy()


def get_cache_status() -> Dict[str, Any]:
    """
    Get current cache status and statistics.

    Returns:
        Dictionary with cache information
    """
    return {
        'populated': _CACHE_POPULATED,
        'canonical_names': len(set(_BUILDING_NAMES_CACHE.values())),
        'aliases': len(_BUILDING_ALIASES_CACHE),
        'metadata_fields_count': sum(len(fields) for fields in _METADATA_FIELDS_CACHE.values()),
        'indexes_with_buildings': _INDEXES_WITH_BUILDINGS.copy()
    }


def clear_building_cache():
    """Clear all building cache data."""
    global _BUILDING_NAMES_CACHE, _BUILDING_ALIASES_CACHE, _METADATA_FIELDS_CACHE
    global _CACHE_POPULATED, _INDEXES_WITH_BUILDINGS

    _BUILDING_NAMES_CACHE.clear()
    _BUILDING_ALIASES_CACHE.clear()
    _METADATA_FIELDS_CACHE.clear()
    _CACHE_POPULATED = False
    _INDEXES_WITH_BUILDINGS.clear()

    logging.info("Building cache cleared")


def get_building_names_from_cache() -> List[str]:
    """
    Get all canonical building names from cache.

    Returns:
        List of canonical building names
    """
    return list(set(_BUILDING_NAMES_CACHE.values()))


# ============================================================================
# FUZZY MATCHING FUNCTIONS
# ============================================================================


def fuzzy_match_score(str1: str, str2: str) -> float:
    """
    Calculate fuzzy match score between two strings.

    Args:
        str1: First string
        str2: Second string

    Returns:
        Similarity score between 0.0 and 1.0
    """
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def find_fuzzy_matches(
    query: str,
    candidates: List[str],
    threshold: float = FUZZY_MATCH_THRESHOLD
) -> List[tuple[str, float]]:
    """
    Find fuzzy matches from a list of candidates.

    Args:
        query: Query string
        candidates: List of candidate strings
        threshold: Minimum similarity threshold (default 0.80)

    Returns:
        List of (candidate, score) tuples sorted by score descending
    """
    query_lower = query.lower().strip()
    matches = []

    for candidate in candidates:
        score = fuzzy_match_score(query_lower, candidate)
        if score >= threshold:
            matches.append((candidate, score))

    # Sort by score descending
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def fuzzy_match_against_metadata_fields(
    query: str,
    canonical_name: str,
    threshold: float = FUZZY_MATCH_THRESHOLD
) -> Optional[tuple[str, float]]:
    """
    Fuzzy match query against all metadata field variations for a building.

    Args:
        query: Query string to match
        canonical_name: Canonical building name
        threshold: Minimum similarity threshold

    Returns:
        (matched_field, score) tuple or None if no match
    """
    if canonical_name not in _METADATA_FIELDS_CACHE:
        return None

    metadata_fields = _METADATA_FIELDS_CACHE[canonical_name]
    matches = find_fuzzy_matches(query, list(metadata_fields), threshold)

    if matches:
        return matches[0]  # Return best match
    return None


# ============================================================================
# BUILDING NAME NORMALIZATION
# ============================================================================


@lru_cache(maxsize=256)
def normalise_building_name(name: str) -> str:
    """
    Normalise building name for matching.

    Args:
        name: Building name

    Returns:
        Normalised building name
    """
    if not name:
        return ""

    # Convert to lowercase
    normalised = name.lower()

    # Remove common suffixes
    suffixes_to_remove = [
        ' building', ' house', ' hall', ' centre', ' center',
        ' complex', ' tower', ' block', ' wing'
    ]
    for suffix in suffixes_to_remove:
        if normalised.endswith(suffix):
            normalised = normalised[:-len(suffix)].strip()

    # Remove extra whitespace
    normalised = ' '.join(normalised.split())

    return normalised


# ============================================================================
# BUILDING EXTRACTION FROM QUERY
# ============================================================================


def extract_building_from_query(
    query: str,
    known_buildings: Optional[List[str]] = None,
    use_cache: bool = True
) -> Optional[str]:
    """
    Extract building name from user query using multiple strategies.
    IMPROVED: Uses fuzzy matching against metadata fields.

    Args:
        query: User query string
        known_buildings: Optional list of known building names
        use_cache: Whether to use cached building data

    Returns:
        Canonical building name if found, None otherwise
    """
    if not query or not query.strip():
        return None

    # Try cache if enabled
    if use_cache and _CACHE_POPULATED:
        known_buildings = get_building_names_from_cache()
    elif use_cache and not _CACHE_POPULATED:
        logging.debug("Building cache not populated, extraction limited")
        return None

    if not known_buildings:
        return None

    # Try patterns to extract potential building name
    extracted_name = None
    for pattern in BUILDING_PATTERNS:
        matches = pattern.findall(query)
        if matches:
            # Get the first match
            candidate = matches[0] if isinstance(
                matches[0], str) else matches[0][0]
            candidate = candidate.strip()

            # Filter out question words and short names
            if (candidate.lower() not in QUESTION_WORDS and
                    len(candidate) >= MIN_BUILDING_NAME_LENGTH):
                extracted_name = candidate
                break

    if not extracted_name:
        logging.debug("No building pattern match found in query")
        return None

    logging.debug("Extracted potential building name: '%s'", extracted_name)

    # Validate against known buildings with fuzzy matching
    return validate_building_name_fuzzy(extracted_name, known_buildings)


def validate_building_name_fuzzy(
    extracted_name: str,
    known_buildings: Optional[List[str]] = None
) -> Optional[str]:
    """
    Validate extracted name against known buildings using fuzzy matching.
    IMPROVED: Checks all metadata field variations with 80% threshold.

    Args:
        extracted_name: Extracted building name candidate
        known_buildings: List of known building names

    Returns:
        Canonical building name if matched, None otherwise
    """
    if not extracted_name:
        return None

    # Use cache if no explicit list provided
    if known_buildings is None:
        if _CACHE_POPULATED:
            known_buildings = get_building_names_from_cache()
        else:
            return None

    if not known_buildings:
        return None

    # Cache the lowercase version for multiple comparisons
    extracted_lower = extracted_name.lower().strip()
    extracted_norm = normalise_building_name(extracted_name)

    # Strategy 1: Exact match in aliases cache
    if _CACHE_POPULATED:
        canonical = _BUILDING_ALIASES_CACHE.get(extracted_lower)
        if canonical:
            logging.info("✅ Alias exact match: '%s' -> '%s'",
                         extracted_name, canonical)
            return canonical

        # Check canonical names cache
        canonical = _BUILDING_NAMES_CACHE.get(extracted_lower)
        if canonical:
            logging.info("✅ Canonical exact match: '%s'", canonical)
            return canonical

    # Strategy 2: Exact match (case-insensitive) in known buildings
    for building in known_buildings:
        if building.lower() == extracted_lower:
            logging.info("✅ Exact match: '%s'", building)
            return building

    # Strategy 3: Substring match (extracted name in building name)
    for building in known_buildings:
        if extracted_lower in building.lower():
            logging.info("✅ Substring match: '%s' in '%s'",
                         extracted_name, building)
            return building

    # Strategy 4: Reverse substring match (building name in extracted name)
    for building in known_buildings:
        if building.lower() in extracted_lower:
            logging.info("✅ Reverse substring match: '%s' contains '%s'",
                         extracted_name, building)
            return building

    # Strategy 5: Fuzzy match against all metadata field variations
    if _CACHE_POPULATED:
        best_match = None
        best_score = 0.0
        best_canonical = None

        for canonical in known_buildings:
            match_result = fuzzy_match_against_metadata_fields(
                extracted_name, canonical, FUZZY_MATCH_THRESHOLD
            )

            if match_result:
                matched_field, score = match_result
                if score > best_score:
                    best_score = score
                    best_match = matched_field
                    best_canonical = canonical

        if best_canonical:
            logging.info(
                "✅ Fuzzy match (%.1f%%): '%s' -> '%s' (via field '%s')",
                best_score * 100,
                extracted_name,
                best_canonical,
                best_match
            )
            return best_canonical

    # Strategy 6: Standard fuzzy match using difflib (80% similarity)
    matches = get_close_matches(
        extracted_name, known_buildings, n=1, cutoff=FUZZY_MATCH_THRESHOLD
    )
    if matches:
        logging.info("✅ Difflib fuzzy match (≥80%%): '%s' -> '%s'",
                     extracted_name, matches[0])
        return matches[0]

    logging.debug("❌ No match found for '%s'", extracted_name)
    return None


# ============================================================================
# RESULT PROCESSING FUNCTIONS
# ============================================================================


def group_results_by_building(
    results: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group search results by normalised building name.
    IMPROVED: Uses enhanced building name extraction from multiple fields.
    """
    grouped = {}

    for result in results:
        # Extract building name from multiple metadata fields
        metadata = result.get('metadata', {})
        building_name = None

        # Check metadata fields in priority order
        for field in ['canonical_building_name', 'building_name', 'Property names', 'UsrFRACondensedPropertyName']:
            value = metadata.get(field) or result.get(field)
            if value:
                if isinstance(value, list):
                    # Take first non-empty item from list
                    for item in value:
                        if item and str(item).strip():
                            building_name = str(item).strip()
                            break
                elif isinstance(value, str) and value.strip():
                    building_name = value.strip()

                if building_name:
                    break

        # Fallback if no building name found
        if not building_name or building_name == 'Unknown':
            building_name = result.get('building_name', 'Unknown')

        normalised_name = normalise_building_name(building_name)

        if not normalised_name:
            normalised_name = building_name

        if normalised_name not in grouped:
            grouped[normalised_name] = []

        if '_normalised_building' not in result:
            result['_normalised_building'] = normalised_name

        # Also store the original building name for reference
        if '_original_building' not in result:
            result['_original_building'] = building_name

        grouped[normalised_name].append(result)

    return grouped


def prioritise_building_results(
    results: List[Dict[str, Any]],
    target_building: str
) -> List[Dict[str, Any]]:
    """
    Reorder results to prioritise a specific building.
    """
    if not target_building or not results:
        return results

    target_normalised = normalise_building_name(target_building).lower()
    target_lower = target_building.lower()

    priority_results = []
    other_results = []

    for result in results:
        building_name = result.get('building_name', '')

        if '_normalised_building' in result:
            normalised = result['_normalised_building'].lower()
        else:
            normalised = normalise_building_name(building_name).lower()

        building_lower = building_name.lower()

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

    return priority_results + other_results


def get_building_context_summary(
    building_results: Dict[str, List[Dict[str, Any]]]
) -> str:
    """
    Create a summary of buildings found in search results.
    """
    if not building_results:
        return ""

    summary_parts = []
    for building, results in building_results.items():
        doc_types = {}
        for r in results:
            doc_type = r.get('document_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

        type_str = ', '.join(
            f"{count} {dtype}" for dtype, count in sorted(doc_types.items())
        )
        summary_parts.append(
            f"- {building}: {len(results)} results ({type_str})"
        )

    return "Buildings found:\n" + "\n".join(summary_parts)


# ============================================================================
# CACHE UTILITIES
# ============================================================================


def clear_caches():
    """Clear all LRU caches."""
    normalise_building_name.cache_clear()
    clear_building_cache()
    logging.info("Cleared building_utils caches")


def get_cache_info() -> Dict[str, Any]:
    """Get information about cache usage for monitoring."""
    return {
        'normalise_building_name': normalise_building_name.cache_info()._asdict(),
        'building_cache': get_cache_status()
    }


# ============================================================================
# DEBUGGING/TESTING UTILITIES
# ============================================================================


def test_building_extraction(test_queries: List[str]) -> Dict[str, Optional[str]]:
    """
    Test building extraction on multiple queries.

    Args:
        test_queries: List of query strings to test

    Returns:
        Dictionary mapping queries to extracted buildings
    """
    results = {}
    for query in test_queries:
        building = extract_building_from_query(query)
        results[query] = building
        logging.info("Test: '%s' -> %s", query, building or "None")

    return results


def get_building_metadata_summary(canonical_name: str) -> Dict[str, Any]:
    """
    Get summary of all metadata field variations for a building.

    Args:
        canonical_name: Canonical building name

    Returns:
        Dictionary with metadata field information
    """
    if canonical_name not in _METADATA_FIELDS_CACHE:
        return {
            'canonical_name': canonical_name,
            'found': False,
            'metadata_fields': []
        }

    return {
        'canonical_name': canonical_name,
        'found': True,
        'metadata_fields': sorted(list(_METADATA_FIELDS_CACHE[canonical_name])),
        'field_count': len(_METADATA_FIELDS_CACHE[canonical_name])
    }
