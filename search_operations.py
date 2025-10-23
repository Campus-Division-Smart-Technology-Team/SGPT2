#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Federated search operations across multiple Pinecone indexes with building-aware search.
IMPROVED VERSION: Enhanced metadata-based building filtering with fuzzy matching.

Key improvements:
- Fuzzy matching (80% threshold) for building filters
- Support for multiple metadata fields (Property names, UsrFRACondensedPropertyName, etc.)
- Better post-query filtering when inference mode doesn't support filters
- Enhanced building detection in search results
"""
import logging
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from heapq import nlargest
import streamlit as st
from clients import oai

from config import (
    TARGET_INDEXES,
    SEARCH_ALL_NAMESPACES,
    DEFAULT_NAMESPACE,
    DEFAULT_EMBED_MODEL,
    SPECIAL_INFERENCE_INDEX,
    SPECIAL_INFERENCE_MODEL,
    MIN_SCORE_THRESHOLD
)
from pinecone_utils import (
    open_index,
    list_namespaces_for_index,
    try_inference_search,
    vector_query,
    normalise_matches,
    embed_texts
)
from answer_generation import (
    enhanced_answer_with_source_date,
    generate_building_focused_answer
)
from date_utils import (
    search_source_for_latest_date,
    parse_date_string,
    format_display_date
)
from building_utils import (
    extract_building_from_query,
    group_results_by_building,
    prioritise_building_results,
    normalise_building_name,
    fuzzy_match_score,
    _BUILDING_ALIASES_CACHE,
    _BUILDING_NAMES_CACHE,
    _METADATA_FIELDS_CACHE,
    _CACHE_POPULATED,
    FUZZY_MATCH_THRESHOLD,
    BUILDING_METADATA_FIELDS
)
from business_terms import BusinessTermMapper

# ============================================================================
# CONSTANTS
# ============================================================================

# Stop words for building name matching
STOP_WORDS = frozenset({
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'bms',
    'building', 'house', 'data', 'planon'
})

# Score boost for matching document types
DOC_TYPE_BOOST_FACTOR = 1.2

# Score boost for matching buildings (higher priority)
BUILDING_BOOST_FACTOR = 2.0  # 2x boost for correct building


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _namespaces_to_search(idx) -> List[str]:
    """Get namespaces to search for given index."""
    if not SEARCH_ALL_NAMESPACES:
        return [DEFAULT_NAMESPACE]
    try:
        return list_namespaces_for_index(idx)
    except RuntimeError:
        return [DEFAULT_NAMESPACE]


def get_doc_type(hit: Dict[str, Any]) -> str:
    """
    Extract document type from hit consistently.
    Checks both metadata and top-level fields.
    """
    metadata = hit.get('metadata', {})
    return metadata.get('document_type') or hit.get('document_type', 'unknown')


# ============================================================================
# ENHANCED BUILDING MATCHING
# ============================================================================


def create_building_metadata_filter(building_filter: str) -> Optional[Dict[str, Any]]:
    """
    Create comprehensive Pinecone metadata filter for building matching.
    IMPROVED: Includes fuzzy matching conditions for all metadata fields.

    Args:
        building_filter: Building name to filter by

    Returns:
        Pinecone filter dictionary or None if no conditions created
    """
    normalise_building = normalise_building_name(building_filter)
    filter_conditions = []

    # Add exact matches for all metadata fields
    for field in BUILDING_METADATA_FIELDS:
        filter_conditions.append({field: {"$eq": building_filter}})
        filter_conditions.append({field: {"$eq": normalise_building}})
        filter_conditions.append({field: {"$eq": building_filter.lower()}})
        filter_conditions.append({field: {"$eq": building_filter.upper()}})

    # Add conditions for known aliases if cache is populated
    if _CACHE_POPULATED and _BUILDING_ALIASES_CACHE:
        # Find all aliases that map to this building
        for alias, canonical in _BUILDING_ALIASES_CACHE.items():
            if canonical == building_filter or canonical.lower() == building_filter.lower():
                for field in BUILDING_METADATA_FIELDS:
                    filter_conditions.append({field: {"$eq": alias}})
                    filter_conditions.append({field: {"$eq": alias.title()}})

    # Add conditions for all known metadata field variations
    if _CACHE_POPULATED and building_filter in _METADATA_FIELDS_CACHE:
        for variation in _METADATA_FIELDS_CACHE[building_filter]:
            for field in BUILDING_METADATA_FIELDS:
                filter_conditions.append({field: {"$eq": variation}})

    # Also try to find the canonical name and add its variations
    if _CACHE_POPULATED:
        canonical = _BUILDING_NAMES_CACHE.get(building_filter.lower())
        if canonical and canonical in _METADATA_FIELDS_CACHE:
            for variation in _METADATA_FIELDS_CACHE[canonical]:
                for field in BUILDING_METADATA_FIELDS:
                    filter_conditions.append({field: {"$eq": variation}})

    # Remove duplicates while preserving order
    seen = set()
    unique_conditions = []
    for condition in filter_conditions:
        # Convert dict to string for comparison
        condition_str = str(sorted(condition.items()))
        if condition_str not in seen:
            seen.add(condition_str)
            unique_conditions.append(condition)

    return {"$or": unique_conditions} if unique_conditions else None


def matches_building_fuzzy(result: Dict[str, Any], target_building: str) -> bool:
    """
    Enhanced building matching with fuzzy matching across all metadata fields.
    IMPROVED: Checks all metadata fields with 80% fuzzy threshold.

    Args:
        result: Search result dictionary
        target_building: Target building name to match

    Returns:
        True if buildings match (exact or fuzzy >= 80%)
    """
    if not target_building:
        return True  # No filter means match everything

    # Extract metadata from result
    metadata = result.get('metadata', {})

    # Normalise target for comparison
    target_norm = normalise_building_name(target_building).lower().strip()
    target_lower = target_building.lower().strip()

    # Check each metadata field
    for field in BUILDING_METADATA_FIELDS:
        field_value = metadata.get(field) or result.get(field)

        if not field_value:
            continue

        # Handle list values
        if isinstance(field_value, list):
            for value in field_value:
                if value and matches_single_building_value(str(value), target_building, target_norm, target_lower):
                    return True

        # Handle string values
        elif isinstance(field_value, str):
            if matches_single_building_value(field_value, target_building, target_norm, target_lower):
                return True

    return False


def matches_single_building_value(
    value: str,
    target_building: str,
    target_norm: str,
    target_lower: str
) -> bool:
    """
    Check if a single building value matches the target.

    Args:
        value: Building value from metadata
        target_building: Original target building name
        target_norm: Normalised target building name
        target_lower: Lowercase target building name

    Returns:
        True if match found
    """
    value_lower = value.lower().strip()
    value_norm = normalise_building_name(value).lower().strip()

    # Strategy 1: Exact match (case-insensitive)
    if value_lower == target_lower:
        return True

    # Strategy 2: Normalised match
    if value_norm == target_norm:
        return True

    # Strategy 3: Substring match
    if target_lower in value_lower or value_lower in target_lower:
        return True

    # Strategy 4: Fuzzy match (80% threshold)
    if fuzzy_match_score(value, target_building) >= FUZZY_MATCH_THRESHOLD:
        return True

    # Strategy 5: Check against cached aliases
    if _CACHE_POPULATED:
        # Check if value is a known alias
        canonical_from_value = _BUILDING_ALIASES_CACHE.get(value_lower)
        canonical_from_target = _BUILDING_ALIASES_CACHE.get(target_lower)

        if canonical_from_value and canonical_from_target:
            if canonical_from_value == canonical_from_target:
                return True

        # Check if either maps to the other's canonical
        if canonical_from_value and canonical_from_value.lower() == target_lower:
            return True
        if canonical_from_target and canonical_from_target.lower() == value_lower:
            return True

    return False


# ============================================================================
# SEARCH OPERATIONS
# ============================================================================


def search_one_index(
    idx_name: str,
    question: str,
    k: int,
    embed_model: Optional[str],
    building_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Query index with optional building metadata filter.
    IMPROVED: Uses enhanced metadata filter with fuzzy matching support.

    Args:
        idx_name: Pinecone index name
        question: Query text
        k: Number of results
        embed_model: Embedding model (None for inference)
        building_filter: Optional building name to filter by

    Returns:
        List of search results
    """
    idx = open_index(idx_name)
    hits: List[Dict[str, Any]] = []

    force_inference = (idx_name == SPECIAL_INFERENCE_INDEX)
    namespaces = _namespaces_to_search(idx)

    for ns in namespaces:
        try:
            # Build Pinecone metadata filter
            pinecone_filter = None
            if building_filter:
                pinecone_filter = create_building_metadata_filter(
                    building_filter)

                if pinecone_filter:
                    logging.info(
                        "Applying building filter for '%s' with %d conditions",
                        building_filter,
                        len(pinecone_filter.get('$or', []))
                    )
                else:
                    logging.warning(
                        "Could not create filter for building '%s'", building_filter)

            # Execute search based on mode
            if force_inference:
                # Inference mode - may not support filters
                raw = try_inference_search(
                    idx, ns, question, k, model_name=SPECIAL_INFERENCE_MODEL
                )
                mode_used = "server-side (inference)"

                # Apply filter post-query if needed
                if building_filter:
                    logging.info(
                        "Inference search doesn't support filters, will apply fuzzy filter post-query"
                    )

            else:
                # Try inference first, fall back to vector query with filter
                try:
                    # Try inference without filter first
                    raw = try_inference_search(
                        idx, ns, question, k * 2, model_name=None)  # Get more results for filtering
                    mode_used = "server-side (inference)"

                    # Will apply filter post-query
                    if building_filter:
                        logging.info(
                            "Will apply fuzzy filter post-query for inference results")

                except Exception:  # pylint: disable=broad-except
                    # Use vector query with filter support
                    vec = embed_texts(
                        [question], embed_model or DEFAULT_EMBED_MODEL)[0]
                    raw = idx.query(
                        vector=vec,
                        top_k=k * 2,  # Get more for post-filtering
                        namespace=ns,
                        filter=pinecone_filter,  # Filter at query time
                        include_metadata=True
                    )
                    mode_used = "client-side (vector + filter)"

            norm = normalise_matches(raw)

            # Post-query fuzzy filtering
            if building_filter:
                original_count = len(norm)
                norm = [
                    r for r in norm
                    if matches_building_fuzzy(r, building_filter)
                ]
                filtered_count = len(norm)
                logging.info(
                    "Post-query fuzzy filter: %d -> %d results (removed %d)",
                    original_count,
                    filtered_count,
                    original_count - filtered_count
                )

            for m in norm:
                m["index"] = idx_name
                m["namespace"] = ns
                m["_mode"] = mode_used

            hits.extend(norm)

        except RuntimeError as e:
            logging.warning("Search failed for %s/%s: %s", idx_name, ns, e)
            continue

    return hits[:k]  # Return only top k after filtering


def matches_building(result_building_name: str, target_building: str) -> bool:
    """
    Enhanced building matching using cached aliases and fuzzy matching.
    IMPROVED: Now uses fuzzy matching across metadata fields.

    Args:
        result_building_name: Building name from search result
        target_building: Target building name to match

    Returns:
        True if buildings match
    """
    # Use the improved fuzzy matching function
    result_dict = {'building_name': result_building_name}
    return matches_building_fuzzy(result_dict, target_building)


def deduplicate_results(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate results based on ID."""
    seen = set()
    unique = []
    for h in hits:
        result_id = h.get('id')
        if result_id and result_id not in seen:
            seen.add(result_id)
            unique.append(h)
    return unique


def get_effective_score(hit: Dict[str, Any]) -> float:
    """Get effective score (boosted if available, otherwise original)."""
    return hit.get('_boosted_score', hit.get('score', 0.0))


def apply_doc_type_boost(
    hits: List[Dict[str, Any]],
    doc_type: str,
    boost_factor: float = DOC_TYPE_BOOST_FACTOR
) -> List[Dict[str, Any]]:
    """
    Apply score boost to results matching document type.

    Args:
        hits: Search results
        doc_type: Document type to boost
        boost_factor: Boost multiplier

    Returns:
        Results with boosted scores
    """
    for hit in hits:
        hit_doc_type = get_doc_type(hit)
        if hit_doc_type == doc_type:
            original_score = hit.get('score', 0.0)
            boosted_score = original_score * boost_factor
            hit['_boosted_score'] = boosted_score
            hit['_doc_type_boost'] = True
            logging.debug(
                "Doc type boost: %.3f -> %.3f for %s",
                original_score,
                boosted_score,
                hit.get('id', 'unknown')
            )

    return hits


def apply_building_boost(
    hits: List[Dict[str, Any]],
    target_building: str,
    boost_factor: float = BUILDING_BOOST_FACTOR
) -> List[Dict[str, Any]]:
    """
    Apply score boost to results matching target building.
    IMPROVED: Uses fuzzy matching to determine building matches.

    Args:
        hits: Search results
        target_building: Target building name
        boost_factor: Boost multiplier

    Returns:
        Results with boosted scores
    """
    boosted_count = 0

    for hit in hits:
        # Use fuzzy matching to check if building matches
        if matches_building_fuzzy(hit, target_building):
            original_score = hit.get('_boosted_score', hit.get('score', 0.0))
            boosted_score = original_score * boost_factor
            hit['_boosted_score'] = boosted_score
            hit['_building_boost'] = True
            boosted_count += 1

            logging.debug(
                "Building boost: %.3f -> %.3f for %s (building: %s)",
                original_score,
                boosted_score,
                hit.get('id', 'unknown'),
                hit.get('building_name', 'unknown')
            )

    logging.info("Applied building boost to %d/%d results",
                 boosted_count, len(hits))
    return hits


def filter_results_by_building(
    hits: List[Dict[str, Any]],
    target_building: str
) -> List[Dict[str, Any]]:
    """
    Filter results to only include target building.
    IMPROVED: Uses fuzzy matching.

    Args:
        hits: Search results
        target_building: Target building name

    Returns:
        Filtered results
    """
    filtered = [
        hit for hit in hits
        if matches_building_fuzzy(hit, target_building)
    ]

    logging.info(
        "Filtered by building '%s': %d -> %d results",
        target_building,
        len(hits),
        len(filtered)
    )

    return filtered


# ============================================================================
# MAIN SEARCH FUNCTION
# ============================================================================


def perform_federated_search(
    query: str,
    top_k: int = 5
) -> Tuple[List[Dict[str, Any]], str, str, bool]:
    """
    Perform federated search with building-aware filtering and business term detection.
    IMPROVED: Uses enhanced fuzzy building matching.

    Args:
        query: User query
        top_k: Number of top results to return

    Returns:
        (results, answer, publication_date_info, score_too_low)
    """
    logging.info("🔍 Federated search: '%s' (k=%d)", query, top_k)

    # ===== QUERY ENHANCEMENT =====
    # Detect building
    target_building = extract_building_from_query(query)
    if target_building:
        logging.info("🏢 Detected building: '%s'", target_building)

    # Detect business terms
    enhanced_query, term_context = BusinessTermMapper.enhance_query_with_terms(
        query)
    doc_type_filter = None

    if term_context:
        detected_terms = list(term_context.keys())
        logging.info("📚 Detected business terms: %s", detected_terms)

        # Get document type from first detected term
        first_term = list(term_context.values())[0]
        doc_type_filter = first_term.get('document_type')
        logging.info("📄 Document type filter: %s", doc_type_filter)

    # ===== TWO-STAGE SEARCH STRATEGY =====
    all_hits = []
    used_filter = False

    # Stage 1: Try filtered search if building detected
    if target_building:
        logging.info(
            "🎯 Stage 1: Filtered search for building '%s'", target_building)

        for idx_name in TARGET_INDEXES:
            hits = search_one_index(
                idx_name,
                enhanced_query,
                top_k * 2,  # Get more results
                embed_model=None,
                building_filter=target_building
            )
            all_hits.extend(hits)

        all_hits = deduplicate_results(all_hits)

        if all_hits:
            used_filter = True
            logging.info(
                "✅ Stage 1 success: %d results with filter", len(all_hits))
        else:
            logging.info(
                "⚠️  Stage 1: No results with filter, proceeding to Stage 2")

    # Stage 2: Unfiltered search with post-query boosting
    if not all_hits:
        logging.info("🌐 Stage 2: Unfiltered search with boosting")

        for idx_name in TARGET_INDEXES:
            hits = search_one_index(
                idx_name,
                enhanced_query,
                top_k * 3,  # Fetch more for better boosting results
                embed_model=None,
                building_filter=None  # No filter in Stage 2
            )
            all_hits.extend(hits)

        all_hits = deduplicate_results(all_hits)

    # Apply document type boosting if applicable
    if doc_type_filter:
        all_hits = apply_doc_type_boost(all_hits, doc_type_filter)

    # Apply building boosting (crucial for Stage 2)
    if target_building:
        all_hits = apply_building_boost(
            all_hits,
            target_building,
            boost_factor=3.0 if not used_filter else 1.5  # Higher boost in Stage 2
        )

    # Sort by effective score (boosted or original)
    all_hits.sort(key=get_effective_score, reverse=True)

    # Get top K results
    top_hits = all_hits[:min(top_k, len(all_hits))]

    logging.info("📊 Returning %d top results (used_filter=%s)",
                 len(top_hits), used_filter)

    # ===== ANSWER GENERATION =====
    answer = ""
    publication_date_info = ""
    score_too_low = False

    # Check score threshold
    if top_hits:
        top_score = get_effective_score(top_hits[0])
        if top_score < MIN_SCORE_THRESHOLD:
            score_too_low = True
            answer = (
                f"I found some results, but they don't seem relevant enough to "
                f"your question. The best matching score was {top_score:.3f}, "
                f"which is below the allowable threshold of {MIN_SCORE_THRESHOLD}. "
                f"Regan has told me to say I don't know. Please try rephrasing "
                f"your question or asking about something else."
            )
            return top_hits, answer, publication_date_info, score_too_low

    # Group results by building
    building_groups = group_results_by_building(top_hits)
    logging.info("Building groups: %s", list(building_groups.keys()))

    # Generate LLM answer if enabled
    if top_hits and st.session_state.get("generate_llm_answer", True):
        if target_building and target_building in building_groups:
            # Building-focused answer
            logging.info(
                "Generating building-focused answer for %s", target_building)
            answer, publication_date_info = generate_building_focused_answer(
                query,
                top_hits[0],
                top_hits,
                target_building,
                building_groups,
                term_context
            )
        else:
            # Standard answer
            logging.info("Generating standard answer")
            answer, publication_date_info = enhanced_answer_with_source_date(
                query,
                top_hits[0],
                top_hits,
                term_context,
                target_building=target_building  # Pass building context
            )

        # Add building summary if multiple buildings found
        if len(building_groups) > 1 and not target_building:
            answer += "\n\n**Note:** Results found across multiple buildings:\n"
            for building, results in list(building_groups.items())[:3]:
                answer += f"\n- **{building}**: {len(results)} result(s)"

    elif top_hits:
        # LLM disabled - get date info
        publication_date_info = _get_publication_date_info(
            top_hits, target_building)

    return top_hits, answer, publication_date_info, score_too_low


def _get_publication_date_info(
    hits: List[Dict[str, Any]],
    target_building: Optional[str] = None
) -> str:
    """
    Extract publication date with building awareness.

    Args:
        hits: Search results
        target_building: Optional building to prioritise

    Returns:
        Formatted date information string
    """
    operational_docs = [
        r for r in hits
        if get_doc_type(r) == 'operational_doc'
    ]

    if not operational_docs:
        return ""

    # Filter by building if specified
    if target_building:
        building_specific = [
            r for r in operational_docs
            if matches_building_fuzzy(r, target_building)
        ]

        if building_specific:
            top_operational = building_specific[0]
            logging.info("Using building-specific doc for date: %s",
                         top_operational.get('building_name'))
        else:
            logging.warning(
                "No operational docs for building '%s'", target_building)
            top_operational = operational_docs[0]
    else:
        top_operational = operational_docs[0]

    key_value = top_operational.get("key", "")
    index_name = top_operational.get("index", "")

    if not key_value or not index_name:
        return ""

    try:
        idx = open_index(index_name)
        latest_date, _ = search_source_for_latest_date(
            idx,
            key_value,
            top_operational.get("namespace", DEFAULT_NAMESPACE)
        )

        if latest_date:
            parsed = parse_date_string(latest_date)
            display_date = format_display_date(parsed)
            return f"📅 Top operational document last updated: **{display_date}**"
        else:
            return "📅 **Publication date unknown** for top operational document"

    except Exception as e:  # pylint: disable=broad-except
        logging.warning("Failed to get publication date: %s", e)
        return ""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def search_by_building(building_name: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Search specifically for all documents related to a building.
    IMPROVED: Uses fuzzy matching filter.

    Args:
        building_name: Name of the building to search for
        top_k: Number of results to return

    Returns:
        List of search results for the building
    """
    all_results = []

    for idx_name in TARGET_INDEXES:
        results = search_one_index(
            idx_name,
            f"building information for {building_name}",
            top_k * 2,
            embed_model=None,
            building_filter=building_name
        )
        all_results.extend(results)

    # Filter by building using fuzzy matching
    filtered_results = filter_results_by_building(all_results, building_name)

    # Sort by effective score
    filtered_results.sort(key=get_effective_score, reverse=True)

    return filtered_results[:top_k]


def get_search_statistics(hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate statistics about search results for debugging/monitoring.

    Args:
        hits: List of search results

    Returns:
        Dictionary with statistics
    """
    doc_type_counts = {}
    building_counts = {}
    index_counts = {}

    for hit in hits:
        # Document types
        doc_type = get_doc_type(hit)
        doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1

        # Buildings
        building = hit.get('building_name', 'Unknown')
        building_counts[building] = building_counts.get(building, 0) + 1

        # Indexes
        index = hit.get('index', 'Unknown')
        index_counts[index] = index_counts.get(index, 0) + 1

    return {
        'total_results': len(hits),
        'doc_types': doc_type_counts,
        'buildings': building_counts,
        'indexes': index_counts,
        'avg_score': sum(hit.get('score', 0) for hit in hits) / len(hits) if hits else 0,
        'max_score': max((hit.get('score', 0) for hit in hits), default=0),
        'min_score': min((hit.get('score', 0) for hit in hits), default=0),
    }
