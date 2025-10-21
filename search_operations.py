#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Federated search operations across multiple Pinecone indexes with building-aware search.
Optimized version with metadata filtering, two-stage search, and building-aware date extraction.
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
    _BUILDING_ALIASES_CACHE,
    _BUILDING_NAMES_CACHE,
    _CACHE_POPULATED
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
                # Normalize building name for better matching
                normalized_building = normalise_building_name(building_filter)

                # Create filter that checks multiple fields
                filter_conditions = [
                    {"building_name": {"$eq": building_filter}},
                    {"building_name": {"$eq": normalized_building}},
                    {"canonical_building_name": {"$eq": building_filter}},
                ]

                # Add alias conditions if available
                if _CACHE_POPULATED and _BUILDING_ALIASES_CACHE:
                    for alias, canonical in _BUILDING_ALIASES_CACHE.items():
                        if canonical == building_filter:
                            filter_conditions.append(
                                {"building_name": {"$eq": alias}})

                pinecone_filter: Union[dict, None] = {"$or": filter_conditions}

                logging.info("Applying building filter: %s", building_filter)

            # Execute search based on mode
            if force_inference:
                # Inference mode - may not support filters
                raw = try_inference_search(
                    idx, ns, question, k, model_name=SPECIAL_INFERENCE_MODEL
                )
                mode_used = "server-side (inference)"

                # Apply filter post-query if needed
                if pinecone_filter:
                    logging.warning(
                        "Inference search doesn't support filters, will filter results post-query"
                    )

            else:
                # Try inference first, fall back to vector query
                try:
                    raw = try_inference_search(
                        idx, ns, question, k, model_name=None)
                    mode_used = "server-side (inference)"
                except Exception:
                    # Use vector query with filter support
                    vec = embed_texts(
                        [question], embed_model or DEFAULT_EMBED_MODEL)[0]
                    raw = idx.query(
                        vector=vec,
                        top_k=k,
                        namespace=ns,
                        filter=pinecone_filter,  # Filter at query time
                        include_metadata=True
                    )
                    mode_used = "client-side (vector + filter)"

            norm = normalise_matches(raw)

            # Post-query filtering if inference was used with filter
            if building_filter and mode_used == "server-side (inference)":
                norm = [
                    r for r in norm
                    if matches_building(r.get('building_name', ''), building_filter)
                ]
                logging.info("Post-query filter: %d results remain", len(norm))

            for m in norm:
                m["index"] = idx_name
                m["namespace"] = ns
                m["_mode"] = mode_used

            hits.extend(norm)

        except RuntimeError as e:
            logging.warning("Search failed for %s/%s: %s", idx_name, ns, e)
            continue

    return hits


def matches_building(result_building_name: str, target_building: str) -> bool:
    """
    Enhanced building matching using cached aliases.

    Args:
        result_building_name: Building name from search result
        target_building: Target building name to match

    Returns:
        True if buildings match
    """
    if not result_building_name or not target_building:
        return False

    # Normalize both
    result_norm = normalise_building_name(result_building_name).lower().strip()
    target_norm = normalise_building_name(target_building).lower().strip()

    # Strategy 1: Exact match after normalization
    if result_norm == target_norm:
        return True

    # Strategy 2: Check if they resolve to same canonical name via cache
    if _CACHE_POPULATED:
        result_canonical = _BUILDING_NAMES_CACHE.get(result_norm) or \
            _BUILDING_ALIASES_CACHE.get(result_norm)
        target_canonical = _BUILDING_NAMES_CACHE.get(target_norm) or \
            _BUILDING_ALIASES_CACHE.get(target_norm)

        if result_canonical and target_canonical and result_canonical == target_canonical:
            return True

    # Strategy 3: Substring matching
    if target_norm in result_norm or result_norm in target_norm:
        return True

    # Strategy 4: Word overlap
    target_words = set(target_norm.split()) - STOP_WORDS
    result_words = set(result_norm.split()) - STOP_WORDS

    if not target_words or not result_words:
        return False

    common_words = target_words & result_words

    if len(common_words) >= 2:
        return True

    if len(target_words) == 1 and len(result_words) == 1 and common_words:
        return True

    return False


def filter_results_by_building(
    results: List[Dict[str, Any]],
    target_building: str
) -> List[Dict[str, Any]]:
    """
    Filter results to only those matching the target building.

    Args:
        results: List of search results
        target_building: Building name to filter by

    Returns:
        Filtered list of results
    """
    if not target_building:
        return results

    filtered = []
    for result in results:
        building_name = result.get('building_name', '')
        if matches_building(building_name, target_building):
            filtered.append(result)

    logging.info(
        "Filtered %d/%d results for building '%s'",
        len(filtered), len(results), target_building
    )

    return filtered


def deduplicate_results(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate results based on ID.

    Args:
        hits: List of search results

    Returns:
        Deduplicated list
    """
    seen_ids: Set[str] = set()
    unique_hits = []

    for hit in hits:
        hit_id = hit.get('id')
        if hit_id and hit_id not in seen_ids:
            seen_ids.add(hit_id)
            unique_hits.append(hit)

    logging.info("Deduplicated %d results to %d unique",
                 len(hits), len(unique_hits))
    return unique_hits


def apply_building_boost(
    hits: List[Dict[str, Any]],
    target_building: str,
    boost_factor: float = BUILDING_BOOST_FACTOR
) -> List[Dict[str, Any]]:
    """
    Apply score boost to results matching the target building.

    Args:
        hits: List of search results
        target_building: Building name to boost
        boost_factor: Multiplier for matching buildings

    Returns:
        Results with building boost metadata added
    """
    if not target_building:
        return hits

    boosted_count = 0

    for hit in hits:
        building_name = hit.get('building_name', '')
        original_score = hit.get('score', 0)

        # Check if this result matches the target building
        is_match = matches_building(building_name, target_building)

        if is_match:
            hit['_building_boost'] = boost_factor
            hit['_boosted_score'] = original_score * boost_factor
            boosted_count += 1
            logging.debug(
                "Boosted '%s' score: %.3f → %.3f",
                building_name, original_score, hit['_boosted_score']
            )
        else:
            # Preserve any existing boost or use original score
            hit['_boosted_score'] = hit.get('_boosted_score', original_score)

    logging.info(
        "Applied building boost (%.1fx) to %d/%d results for '%s'",
        boost_factor, boosted_count, len(hits), target_building
    )

    return hits


def apply_doc_type_boost(
    hits: List[Dict[str, Any]],
    doc_type_filter: List[str]
) -> List[Dict[str, Any]]:
    """
    Apply score boost to results matching target document types.
    Creates new boosted score without mutating original.

    Args:
        hits: List of search results
        doc_type_filter: List of document types to boost

    Returns:
        Results with boost metadata added
    """
    if not doc_type_filter:
        return hits

    boosted_count = 0
    for hit in hits:
        doc_type = get_doc_type(hit)
        original_score = hit.get('score', 0)

        if doc_type in doc_type_filter:
            hit['_doc_type_boost'] = DOC_TYPE_BOOST_FACTOR
            hit['_boosted_score'] = original_score * DOC_TYPE_BOOST_FACTOR
            boosted_count += 1
        else:
            hit['_boosted_score'] = original_score

    logging.info("Applied doc type boost to %d/%d results",
                 boosted_count, len(hits))
    return hits


def get_effective_score(hit: Dict[str, Any]) -> float:
    """Get the effective score for a hit (boosted if available, otherwise original)."""
    return hit.get('_boosted_score', hit.get('score', 0))


# ============================================================================
# MAIN SEARCH FUNCTION
# ============================================================================


def perform_federated_search(
    query: str,
    top_k: int
) -> Tuple[List[Dict[str, Any]], str, str, bool]:
    """
    Enhanced federated search with two-stage building-aware approach.

    Stage 1: If building detected, search with metadata filter
    Stage 2: If insufficient results, fall back to semantic search + boosting

    Args:
        query: User query string
        top_k: Number of top results to return

    Returns:
        (results, answer, publication_date_info, score_too_low)
    """
    # Extract building name from query
    target_building = extract_building_from_query(
        query, use_cache=_CACHE_POPULATED)

    # Detect and enhance query with business terms
    enhanced_query, term_context = BusinessTermMapper.enhance_query_with_terms(
        query)

    # Determine document type filter
    doc_type_filter = None
    if term_context:
        doc_types = {info['document_type'] for info in term_context.values()}
        doc_type_filter = list(doc_types)
        logging.info("Detected business terms: %s", list(term_context.keys()))
        logging.info("Filtering for document types: %s", doc_type_filter)

    # Log detection results
    if target_building:
        logging.info("🏢 Detected building: %s", target_building)
        st.info(f"🏢 Detected building: **{target_building}**")

    if term_context:
        terms_str = ', '.join(
            f"**{t}** ({info['full_name']})"
            for t, info in term_context.items()
        )
        st.info(f"📄 Detected terms: {terms_str}")

    # ===== STAGE 1: FILTERED SEARCH (if building detected) =====
    all_hits = []
    used_filter = False

    if target_building:
        logging.info(
            "🔍 Stage 1: Filtered search for building '%s'", target_building)

        for idx_name in TARGET_INDEXES:
            hits = search_one_index(
                idx_name,
                enhanced_query,
                top_k * 2,
                embed_model=None,
                building_filter=target_building  # Use metadata filter
            )
            all_hits.extend(hits)

        all_hits = deduplicate_results(all_hits)

        # Check if we got enough quality results
        if len(all_hits) >= top_k and all_hits[0].get('score', 0) > MIN_SCORE_THRESHOLD:
            logging.info("✅ Stage 1 successful: %d results (top score: %.3f)",
                         len(all_hits), all_hits[0].get('score', 0))
            used_filter = True
        else:
            logging.warning(
                "⚠️ Stage 1 insufficient: %d results (top score: %.3f), proceeding to Stage 2",
                len(all_hits),
                all_hits[0].get('score', 0) if all_hits else 0
            )
            all_hits = []  # Clear and retry with semantic search

    # ===== STAGE 2: SEMANTIC SEARCH + BOOSTING (fallback or no building) =====
    if not used_filter:
        logging.info("🔍 Stage 2: Semantic search%s",
                     " with building boosting" if target_building else "")

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
        target_building: Optional building to prioritize

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
            if matches_building(r.get('building_name', ''), target_building)
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

    except Exception as e:
        logging.warning("Failed to get publication date: %s", e)
        return ""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def search_by_building(building_name: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Search specifically for all documents related to a building.

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

    # Filter by building
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
