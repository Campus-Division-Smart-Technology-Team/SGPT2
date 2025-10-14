#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Federated search operations across multiple Pinecone indexes with building-aware search.
"""

import streamlit as st
import logging
from typing import Dict, List, Optional, Any, Tuple
from heapq import nlargest

from config import (TARGET_INDEXES, SEARCH_ALL_NAMESPACES, DEFAULT_NAMESPACE,
                    SPECIAL_INFERENCE_INDEX, SPECIAL_INFERENCE_MODEL, MIN_SCORE_THRESHOLD)
from pinecone_utils import (open_index, list_namespaces_for_index,
                            try_inference_search, vector_query, normalise_matches)
from answer_generation import enhanced_answer_with_source_date, generate_building_focused_answer
from date_utils import search_source_for_latest_date, parse_date_string, format_display_date
from building_utils import (extract_building_from_query, group_results_by_building,
                            prioritise_building_results, get_building_context_summary)


def _namespaces_to_search(idx):
    """Get namespaces to search for given index."""
    if not SEARCH_ALL_NAMESPACES:
        return [DEFAULT_NAMESPACE]
    try:
        return list_namespaces_for_index(idx)
    except Exception:
        return [DEFAULT_NAMESPACE]


def embed_texts(texts: List[str], model: str) -> List[List[float]]:
    """Generate embeddings for texts (helper for vector search with filters)."""
    from clients import oai
    res = oai.embeddings.create(model=model, input=texts)
    return [d.embedding for d in res.data]


def search_one_index(idx_name: str, question: str, k: int, embed_model: Optional[str],
                     building_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Query a single index across its namespaces with optional building filter.
    """
    idx = open_index(idx_name)
    hits: List[Dict[str, Any]] = []

    force_inference = (idx_name == SPECIAL_INFERENCE_INDEX)
    namespaces = _namespaces_to_search(idx)

    for ns in namespaces:
        try:
            if force_inference:
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
                    raw = vector_query(idx, ns, question, k,
                                       embed_model or DEFAULT_EMBED_MODEL)
                    mode_used = "client-side (vector)"

            norm = normalise_matches(raw)

            for m in norm:
                m["index"] = idx_name
                m["namespace"] = ns
                m["_mode"] = mode_used
            hits.extend(norm)

        except Exception as e:
            logging.warning(f"Search failed for {idx_name}/{ns}: {e}")
            continue

    return hits


def matches_building(result_building_name: str, target_building: str) -> bool:
    """
    Check if a result's building name matches the target building.
    Uses flexible matching to handle variations.
    """
    if not result_building_name or not target_building:
        return False

    result_lower = result_building_name.lower().strip()
    target_lower = target_building.lower().strip()

    # Exact match
    if result_lower == target_lower:
        return True

    # Target is contained in result
    if target_lower in result_lower:
        return True

    # Result is contained in target
    if result_lower in target_lower:
        return True

    # Check if they share significant words
    target_words = set(target_lower.split())
    result_words = set(result_lower.split())

    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'bms',
                  'building', 'house', 'data', 'planon'}

    target_words = target_words - stop_words
    result_words = result_words - stop_words

    common_words = target_words & result_words

    if len(common_words) >= 2:
        return True

    if len(target_words) >= 1 and len(result_words) >= 1 and common_words:
        return True

    return False


def filter_results_by_building(results: List[Dict[str, Any]],
                               target_building: str) -> List[Dict[str, Any]]:
    """Filter results to only those matching the target building."""
    filtered = []

    for result in results:
        building_name = result.get('building_name', '')
        if matches_building(building_name, target_building):
            filtered.append(result)

    return filtered


def perform_federated_search(query: str, top_k: int) -> Tuple[List[Dict[str, Any]], str, str, bool]:
    """
    Perform federated search with building-aware prioritization.

    Returns: (results, answer, publication_date_info, score_too_low)
    """
    # Extract building name from query if present
    target_building = extract_building_from_query(query)

    if target_building:
        logging.info("🏢 Detected building: %s", target_building)
        st.info(
            f"🏢 Detected building: **{target_building}** - searching for all related documents")

    all_hits: List[Dict[str, Any]] = []

    # Search across all target indexes
    for idx_name in TARGET_INDEXES:
        # If we have a target building, enhance the query
        if target_building:
            enhanced_query = f"{query} {target_building}"
            building_hits = search_one_index(
                idx_name, enhanced_query, top_k * 2, embed_model=None)
            all_hits.extend(building_hits)

        # Also do a standard search
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

    logging.info("Total unique hits before filtering: %d", len(unique_hits))

    # DEBUG: Log document types found
    doc_type_counts = {}
    for hit in unique_hits:
        doc_type = hit.get('document_type', 'unknown')
        doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
    logging.info("Document types found: %s", doc_type_counts)

    # DEBUG: Log top 5 results before filtering
    logging.info("Top 5 results before filtering:")
    for i, hit in enumerate(unique_hits[:5]):
        logging.info(
            "  %d. building_name='%s', doc_type='%s', score=%.3f, key='%s'",
            i+1,
            hit.get('building_name'),
            hit.get('document_type'),
            hit.get('score', 0),
            hit.get('key', '')[:50]
        )
    # If we have a target building, filter and prioritise
    if target_building:
        building_specific_hits = filter_results_by_building(
            unique_hits, target_building)

        logging.info("Hits matching '%s': %d", target_building,
                     len(building_specific_hits))

        # DEBUG: Log document types in filtered results
        filtered_doc_type_counts = {}
        for hit in building_specific_hits:
            doc_type = hit.get('document_type', 'unknown')
            filtered_doc_type_counts[doc_type] = filtered_doc_type_counts.get(
                doc_type, 0) + 1
        logging.info("Filtered document types: %s", filtered_doc_type_counts)

        # Log what we found
        logging.info("Top results after filtering:")
        for i, hit in enumerate(building_specific_hits[:5]):
            logging.info(
                "  %d. building_name='%s', doc_type='%s', score=%.3f",
                i+1,
                hit.get('building_name'),
                hit.get('document_type'),
                hit.get('score', 0))
        # If we found any matches, use them; otherwise, fall back to prioritised full list
        if building_specific_hits:
            unique_hits = building_specific_hits
        else:
            logging.warning(
                "No exact matches for '%s', using all results", target_building)
            unique_hits = prioritise_building_results(
                unique_hits, target_building)

    # Get top K results by score
    top_hits = nlargest(min(top_k, len(unique_hits)),
                        unique_hits, key=lambda m: (m.get("score") or 0))

    # DEBUG: Log final top results
    logging.info("Final top %d results:", len(top_hits))
    for i, hit in enumerate(top_hits):
        logging.info(
            "  %d. building_name='%s', doc_type='%s', score=%.3f, key='%s'",
            i+1,
            hit.get('building_name'),
            hit.get('document_type'),
            hit.get('score', 0),
            hit.get('key', '')[:50])

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
    # building_summary = get_building_context_summary(building_groups)

    logging.info("Building groups: %s", list(building_groups.keys()))

    if top_hits and st.session_state.get("generate_llm_answer", True):
        # If query is building-focused, use specialised answer generation
        if target_building and target_building in building_groups:
            logging.info(
                "Generating building-focused answer for %s", target_building)
            answer, publication_date_info = generate_building_focused_answer(
                query, top_hits[0], top_hits, target_building, building_groups
            )
        else:
            # Use standard answer generation
            logging.info("Generating standard answer")
            answer, publication_date_info = enhanced_answer_with_source_date(
                query, top_hits[0], top_hits
            )

        # Add building summary if multiple buildings found and not targeting specific building
        if len(building_groups) > 1 and not target_building:
            answer += "\n\n**Note:** Results found across multiple buildings:\n"
            for building, results in list(building_groups.items())[:3]:
                answer += f"\n- **{building}**: {len(results)} result(s)"

    elif top_hits:
        # If LLM answer generation is disabled, still try to get date info
        # Find the highest-scoring operational doc
        operational_docs = [r for r in top_hits if r.get(
            'document_type') == 'operational_doc']

        if operational_docs:
            top_operational = operational_docs[0]  # Already sorted by score
            key_value = top_operational.get("key", "")

            index_name = top_operational.get("index") or ""
            if key_value and index_name:
                idx = open_index(index_name)
                latest_date, _ = search_source_for_latest_date(
                    idx, key_value, top_operational.get(
                        "namespace", DEFAULT_NAMESPACE)
                )
                if latest_date:
                    parsed = parse_date_string(latest_date)
                    display_date = format_display_date(parsed)
                    publication_date_info = f"📅 Top operational document last updated: **{display_date}**"
                else:
                    publication_date_info = f"📅 **Publication date unknown** for top operational document"

    return top_hits, answer, publication_date_info, score_too_low


def search_by_building(building_name: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Search specifically for all documents related to a building.
    """
    all_results = []

    for idx_name in TARGET_INDEXES:
        results = search_one_index(
            idx_name,
            f"building information for {building_name}",
            top_k * 2,
            embed_model=None
        )
        all_results.extend(results)

    # Filter by building
    filtered_results = filter_results_by_building(all_results, building_name)

    # Sort by score
    filtered_results.sort(key=lambda x: x.get('score', 0), reverse=True)

    return filtered_results[:top_k]
