#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Federated search operations across multiple Pinecone indexes with building-aware search.
"""

from typing import Dict, List, Optional, Any, Tuple
from heapq import nlargest
import streamlit as st

from config import (
    TARGET_INDEXES, SEARCH_ALL_NAMESPACES, DEFAULT_NAMESPACE,
    SPECIAL_INFERENCE_INDEX, SPECIAL_INFERENCE_MODEL, MIN_SCORE_THRESHOLD
)
from pinecone_utils import (
    open_index, list_namespaces_for_index, try_inference_search,
    vector_query, normalize_matches
)
from answer_generation import enhanced_answer_with_source_date, generate_building_focused_answer
from date_utils import search_source_for_latest_date, parse_date_string, format_display_date
from building_utils import (
    extract_building_from_query, group_results_by_building,
    prioritize_building_results, get_building_context_summary
)


def _namespaces_to_search(idx):
    """Get namespaces to search for given index."""
    if not SEARCH_ALL_NAMESPACES:
        return [DEFAULT_NAMESPACE]
    try:
        return list_namespaces_for_index(idx)
    except Exception:
        return [DEFAULT_NAMESPACE]


def search_one_index(idx_name: str, question: str, k: int, embed_model: Optional[str],
                     building_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Query a single index across its namespaces with optional building filter.

    Args:
        idx_name: Name of the Pinecone index
        question: Search query
        k: Number of results to return
        embed_model: Embedding model to use
        building_filter: Optional building name to filter results

    Returns:
        List of search results with metadata
    """
    idx = open_index(idx_name)
    hits: List[Dict[str, Any]] = []

    force_inference = (idx_name == SPECIAL_INFERENCE_INDEX)
    namespaces = _namespaces_to_search(idx)

    for ns in namespaces:
        try:
            # Build Pinecone metadata filter if building is specified
            metadata_filter = None
            if building_filter:
                # Create a filter that matches building_name
                metadata_filter = {
                    "$or": [
                        {"building_name": {"$eq": building_filter}},
                        # Case-insensitive partial match
                        {"building_name": {
                            "$regex": f"(?i){building_filter}"}},
                    ]
                }

            if force_inference:
                # Note: Pinecone inference search may not support filters in all cases
                raw = try_inference_search(
                    idx, ns, question, k, model_name=SPECIAL_INFERENCE_MODEL)
                mode_used = "server-side (inference)"
            else:
                try:
                    raw = try_inference_search(
                        idx, ns, question, k, model_name=None)
                    mode_used = "server-side (inference)"
                except Exception:
                    from config import DEFAULT_EMBED_MODEL
                    # Vector query supports metadata filters
                    if metadata_filter:
                        raw = idx.query(
                            vector=embed_texts(
                                [question], embed_model or DEFAULT_EMBED_MODEL)[0],
                            top_k=k,
                            namespace=ns,
                            filter=metadata_filter,
                            include_metadata=True
                        )
                    else:
                        raw = vector_query(
                            idx, ns, question, k, embed_model or DEFAULT_EMBED_MODEL)
                    mode_used = "client-side (vector)"

            norm = normalize_matches(raw)

            # Post-filter results if inference search was used (can't use metadata filters)
            if building_filter and force_inference:
                norm = [m for m in norm if building_filter.lower() in m.get(
                    'building_name', '').lower()]

            for m in norm:
                m["index"] = idx_name
                m["namespace"] = ns
                m["_mode"] = mode_used
            hits.extend(norm)

        except Exception as e:
            # non-fatal per-namespace failure
            import logging
            logging.warning(f"Search failed for {idx_name}/{ns}: {e}")
            continue

    return hits


def embed_texts(texts: List[str], model: str) -> List[List[float]]:
    """Generate embeddings for texts (helper for vector search with filters)."""
    from clients import oai
    res = oai.embeddings.create(model=model, input=texts)
    return [d.embedding for d in res.data]


def perform_federated_search(query: str, top_k: int) -> Tuple[List[Dict[str, Any]], str, str, bool]:
    """
    Perform federated search with building-aware prioritization.

    Returns: (results, answer, publication_date_info, score_too_low)
    """
    # Extract building name from query if present
    target_building = extract_building_from_query(query)

    if target_building:
        st.info(
            f"🏢 Detected building: **{target_building}** - prioritizing results for this building")

    all_hits: List[Dict[str, Any]] = []

    # Search across all target indexes
    for idx_name in TARGET_INDEXES:
        # First search: try with building filter if we detected one
        if target_building:
            building_hits = search_one_index(idx_name, query, top_k, embed_model=None,
                                             building_filter=target_building)
            all_hits.extend(building_hits)

        # Second search: general search without filter (get more results)
        general_hits = search_one_index(
            idx_name, query, top_k, embed_model=None)
        all_hits.extend(general_hits)

    # Remove duplicates based on ID
    seen_ids = set()
    unique_hits = []
    for hit in all_hits:
        hit_id = hit.get('id')
        if hit_id and hit_id not in seen_ids:
            seen_ids.add(hit_id)
            unique_hits.append(hit)

    # Prioritize results from target building if specified
    if target_building:
        unique_hits = prioritize_building_results(unique_hits, target_building)

    # Get top K results by score
    top_hits = nlargest(top_k, unique_hits,
                        key=lambda m: (m.get("score") or 0))

    answer = ""
    publication_date_info = ""
    score_too_low = False

    # Check if top score is below threshold
    if top_hits:
        top_score = top_hits[0].get("score", 0)
        if top_score < MIN_SCORE_THRESHOLD:
            score_too_low = True
            answer = f"I found some results, but they don't seem relevant enough to your question. The best matching score was {top_score:.3f}, which is below the allowable threshold of {MIN_SCORE_THRESHOLD}. Regan has told me to say I don't know. Please try rephrasing your question or asking about something else."
            return top_hits, answer, publication_date_info, score_too_low

    # Group results by building for better context
    building_groups = group_results_by_building(top_hits)
    building_summary = get_building_context_summary(building_groups)

    if top_hits and st.session_state.get("generate_llm_answer", True):
        # If query is building-focused, use specialized answer generation
        if target_building and len(building_groups) > 0:
            answer, publication_date_info = generate_building_focused_answer(
                query, top_hits[0], top_hits, target_building, building_groups
            )
        else:
            # Use standard answer generation
            answer, publication_date_info = enhanced_answer_with_source_date(
                query, top_hits[0], top_hits
            )

        # Add building summary if multiple buildings found
        if len(building_groups) > 1 and not target_building:
            answer += f"\n\n**Note:** Results found across multiple buildings:\n"
            # Show top 3
            for building, results in list(building_groups.items())[:3]:
                answer += f"\n- **{building}**: {len(results)} result(s)"

    elif top_hits:
        # If LLM answer generation is disabled, still try to get date info
        top_result = top_hits[0]
        key_value = top_result.get("key", "")
        if key_value and top_result.get("index"):
            idx = open_index(top_result.get("index"))
            latest_date, _ = search_source_for_latest_date(
                idx, key_value, top_result.get("namespace", DEFAULT_NAMESPACE)
            )
            if latest_date:
                parsed = parse_date_string(latest_date)
                display_date = format_display_date(parsed)
                publication_date_info = f"📅 Top result document last updated: **{display_date}**"
            else:
                publication_date_info = f"📅 **Publication date unknown** for top result"

    return top_hits, answer, publication_date_info, score_too_low


def search_by_building(building_name: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Search specifically for all documents related to a building.

    Args:
        building_name: Name of the building
        top_k: Number of results per index

    Returns:
        List of all results for the building
    """
    all_results = []

    for idx_name in TARGET_INDEXES:
        results = search_one_index(
            idx_name,
            f"building information for {building_name}",
            top_k,
            embed_model=None,
            building_filter=building_name
        )
        all_results.extend(results)

    # Sort by score
    all_results.sort(key=lambda x: x.get('score', 0), reverse=True)

    return all_results
