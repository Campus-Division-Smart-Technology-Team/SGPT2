#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Federated search operations across multiple Pinecone indexes.
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
from answer_generation import enhanced_answer_with_source_date
from date_utils import search_source_for_latest_date, parse_date_string, format_display_date


def _namespaces_to_search(idx):
    """Get namespaces to search for given index."""
    if not SEARCH_ALL_NAMESPACES:
        return [DEFAULT_NAMESPACE]
    try:
        return list_namespaces_for_index(idx)
    except Exception:
        return [DEFAULT_NAMESPACE]


def search_one_index(idx_name: str, question: str, k: int, embed_model: Optional[str]) -> List[Dict[str, Any]]:
    """Query a single index across its namespaces; attach provenance."""
    idx = open_index(idx_name)
    hits: List[Dict[str, Any]] = []

    force_inference = (idx_name == SPECIAL_INFERENCE_INDEX)
    namespaces = _namespaces_to_search(idx)

    for ns in namespaces:
        try:
            if force_inference:
                raw = try_inference_search(idx, ns, question, k, model_name=SPECIAL_INFERENCE_MODEL)
                mode_used = "server-side (inference)"
            else:
                try:
                    raw = try_inference_search(idx, ns, question, k, model_name=None)
                    mode_used = "server-side (inference)"
                except Exception:
                    from config import DEFAULT_EMBED_MODEL
                    raw = vector_query(idx, ns, question, k, embed_model or DEFAULT_EMBED_MODEL)
                    mode_used = "client-side (vector)"
            norm = normalize_matches(raw)
            for m in norm:
                m["index"] = idx_name
                m["namespace"] = ns
                m["_mode"] = mode_used
            hits.extend(norm)
        except Exception:
            # non-fatal per-namespace failure
            continue

    return hits


def perform_federated_search(query: str, top_k: int) -> Tuple[List[Dict[str, Any]], str, str, bool]:
    """
    Perform federated search and return results, answer, publication date info, and low score flag.
    Returns: (results, answer, publication_date_info, score_too_low)
    """
    all_hits: List[Dict[str, Any]] = []

    for idx_name in TARGET_INDEXES:
        all_hits.extend(search_one_index(idx_name, query, top_k, embed_model=None))

    # Merge & keep a global top_k
    top_hits = nlargest(top_k, all_hits, key=lambda m: (m.get("score") or 0))

    answer = ""
    publication_date_info = ""
    score_too_low = False

    # Check if top score is below threshold
    if top_hits:
        top_score = top_hits[0].get("score", 0)
        if top_score < MIN_SCORE_THRESHOLD:
            score_too_low = True
            answer = f"I found some results, but they don't seem relevant enough to your question (best match score: {top_score:.3f}, threshold: {MIN_SCORE_THRESHOLD}). Regan has told me to say I don't know. Please try rephrasing your question or asking about something else."
            return top_hits, answer, publication_date_info, score_too_low

    if top_hits and st.session_state.get("generate_llm_answer", True):
        # Use enhanced answer generation with source date search
        answer, publication_date_info = enhanced_answer_with_source_date(query, top_hits[0], top_hits)
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
