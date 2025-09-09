#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Answer generation using OpenAI with enhanced source date information.
"""

from typing import Dict, List, Tuple, Any
from clients import oai
from config import ANSWER_MODEL, DEFAULT_NAMESPACE
from date_utils import search_source_for_latest_date, parse_date_string, format_display_date
from pinecone_utils import open_index


def build_context(snippets: List[str], max_chars: int = 6000) -> str:
    """Build context string from text snippets."""
    blob = ("\n\n".join(s.strip() for s in snippets if s.strip()))
    return blob if len(blob) <= max_chars else blob[:max_chars]


def enhanced_answer_with_source_date(question: str, top_result: Dict[str, Any], all_results: List[Dict[str, Any]]) -> \
Tuple[str, str]:
    """
    Generate an answer that includes information about the source's latest publication date.
    Returns: (answer, publication_date_info)
    """
    # Get the key from top result
    key_value = top_result.get("key", "")
    idx_name = top_result.get("index", "")

    latest_date = None
    source_doc_count = 1

    if key_value and idx_name:
        idx = open_index(idx_name)
        latest_date, source_docs = search_source_for_latest_date(
            idx, key_value, top_result.get("namespace", DEFAULT_NAMESPACE)
        )
        source_doc_count = len([d for d in source_docs if d.get("key") == key_value])

    # Format date information
    if latest_date:
        parsed = parse_date_string(latest_date)
        display_date = format_display_date(parsed)
        date_context = f"The document '{key_value}' was last updated/reviewed on {display_date}."
        publication_info = f"📅 Document last updated: **{display_date}** (from {key_value}, searched {source_doc_count} related chunks)"
    else:
        date_context = f"The publication date for document '{key_value}' could not be determined."
        publication_info = f"📅 **Publication date unknown** (document: {key_value})"

    # Build context from all results
    snippets = [r.get("text", "") for r in all_results if r.get("text")]
    context = build_context(snippets)

    prompt = f"""Your name is Alfred, a helpful assistant at the University of Bristol working in the Smart Technology team.

Answer the user's question using ONLY the context below. If the answer cannot be found in the context, tell the user that Regan has told you to say you don't know.

IMPORTANT: Please ensure that you always end your response with information about when the document was last updated or published.

Question: {question}

Context: {context}

Document Information: {date_context}

Top Result Details:
- Document: {key_value}
- Score: {top_result.get('score', 'Unknown'):.3f}
- Index: {idx_name}
"""

    chat = oai.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system",
             "content": "You are Alfred, a helpful assistant. Always include document date information in your responses."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    answer = chat.choices[0].message.content.strip()
    return answer, publication_info
