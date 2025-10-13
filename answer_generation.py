#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Answer generation using OpenAI with enhanced source date information and building-aware context.
"""

from typing import Dict, List, Tuple, Any
from clients import oai
from config import ANSWER_MODEL, DEFAULT_NAMESPACE
from date_utils import search_source_for_latest_date, parse_date_string, format_display_date
from pinecone_utils import open_index
import logging


def build_context(snippets: List[str], max_chars: int = 6000,
                  prioritize_building: bool = False) -> str:
    """
    Build context string from text snippets with optional building prioritization.

    Args:
        snippets: List of text snippets to include
        max_chars: Maximum characters for context
        prioritize_building: If True, prioritize property/building data

    Returns:
        Context string for LLM
    """
    if prioritize_building:
        # Separate property data from operational docs
        property_snippets = []
        operational_snippets = []

        for snippet in snippets:
            # Property data starts with "Building:"
            if snippet and 'Building:' in snippet[:100]:
                property_snippets.append(snippet)
            else:
                operational_snippets.append(snippet)

        # Combine with property data first
        snippets = property_snippets + operational_snippets

    blob = "\n\n---\n\n".join(s.strip() for s in snippets if s and s.strip())
    return blob if len(blob) <= max_chars else blob[:max_chars]


def build_building_grouped_context(results: List[Dict[str, Any]],
                                   max_chars: int = 6000) -> str:
    """
    Build context grouped by building for better organization.

    Args:
        results: Search results with building_name metadata
        max_chars: Maximum characters for context

    Returns:
        Organized context string
    """
    building_groups = {}

    # Group by building
    for result in results:
        building = result.get('building_name', 'Unknown Building')
        if building not in building_groups:
            building_groups[building] = []
        building_groups[building].append(result)

    # Build context with building sections
    context_parts = []
    char_count = 0

    for building, building_results in building_groups.items():
        section = f"=== {building} ===\n\n"

        for result in building_results:
            text = result.get('text', '')
            doc_type = result.get('document_type', 'unknown')
            key = result.get('key', '')

            snippet = f"[{doc_type}] {key}:\n{text}\n"

            if char_count + len(section) + len(snippet) > max_chars:
                break

            section += snippet + "\n"
            char_count += len(snippet)

        context_parts.append(section)

        if char_count >= max_chars:
            break

    return "\n".join(context_parts)


def enhanced_answer_with_source_date(question: str, top_result: Dict[str, Any],
                                     all_results: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Generate an answer that includes information about the source's latest publication date.

    Returns: (answer, publication_date_info)
    """
    # Get the key from top result
    key_value = top_result.get("key", "")
    idx_name = top_result.get("index", "")
    source_value = top_result.get("source", "")
    building_name = top_result.get("building_name", "Unknown")

    latest_date = None
    source_doc_count = 1

    if key_value and idx_name:
        idx = open_index(idx_name)
        latest_date, source_docs = search_source_for_latest_date(
            idx, key_value, top_result.get("namespace", DEFAULT_NAMESPACE)
        )
        source_doc_count = len(
            [d for d in source_docs if d.get("key") == key_value])

    # Format date information
    if latest_date:
        parsed = parse_date_string(latest_date)
        display_date = format_display_date(parsed)
        date_context = f"The document '{key_value}' was last updated/reviewed on {display_date}."
        publication_info = f"📅 Document last updated: **{display_date}** (from {key_value}, searched {source_doc_count} related chunks)"
    else:
        date_context = f"The publication date for document '{key_value}' could not be determined."
        publication_info = f"📅 **Publication date unknown** ({source_value})"

    # Build context with building prioritization
    snippets = [r.get("text", "") for r in all_results if r.get("text")]
    context = build_context(snippets, prioritize_building=True)

    prompt = f"""Your name is Alfred, a helpful assistant at the University of Bristol working in the Smart Technology team.

Answer the user's question using ONLY the context below. If the answer cannot be found in the context, tell the user that Regan has told you to say you don't know.

IMPORTANT: 
- Always end your response with information about when the document was last updated or published.
- If the query relates to a specific building, make sure to mention the building name in your answer.

Question: {question}

Context: {context}

Document Information: {date_context}

Top Result Details:
- Building: {building_name}
- Document: {key_value}
- Score: {top_result.get('score', 'Unknown'):.3f}
- Index: {idx_name}
"""

    try:
        chat = oai.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[
                {"role": "system",
                 "content": "You are Alfred, a helpful assistant at the University of Bristol. Always include document date information in your responses. When answering about buildings, clearly state which building the information relates to."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        answer = chat.choices[0].message.content.strip()
        return answer, publication_info
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        return "I encountered an error generating the answer. Please try again.", publication_info


def generate_building_focused_answer(question: str, top_result: Dict[str, Any],
                                     all_results: List[Dict[str, Any]],
                                     target_building: str,
                                     building_groups: Dict[str, List[Dict[str, Any]]]) -> Tuple[str, str]:
    """
    Generate an answer specifically focused on a particular building with grouped context.

    Args:
        question: User's question
        top_result: Highest scoring result
        all_results: All search results
        target_building: The building name to focus on
        building_groups: Results grouped by building

    Returns: (answer, publication_date_info)
    """
    # Get results specific to target building
    target_results = building_groups.get(target_building, [])

    if not target_results:
        # Fallback to standard answer generation
        return enhanced_answer_with_source_date(question, top_result, all_results)

    # Separate by document type
    property_data = [r for r in target_results if r.get(
        'document_type') == 'planon_data']
    operational_docs = [r for r in target_results if r.get(
        'document_type') == 'operational_doc']

    # Build grouped context
    context = build_building_grouped_context(target_results, max_chars=6000)

    # Get date information from top result
    key_value = top_result.get("key", "")
    idx_name = top_result.get("index", "")

    latest_date = None
    if key_value and idx_name:
        idx = open_index(idx_name)
        latest_date, _ = search_source_for_latest_date(
            idx, key_value, top_result.get("namespace", DEFAULT_NAMESPACE)
        )

    if latest_date:
        parsed = parse_date_string(latest_date)
        display_date = format_display_date(parsed)
        date_context = f"Most recent document update: {display_date}"
        publication_info = f"📅 Latest update for {target_building}: **{display_date}**"
    else:
        date_context = "Document dates not available"
        publication_info = f"📅 **Publication date unknown** for {target_building}"

    # Create document type summary
    doc_summary = []
    if property_data:
        doc_summary.append(f"{len(property_data)} property data record(s)")
    if operational_docs:
        doc_summary.append(f"{len(operational_docs)} operational document(s)")

    doc_summary_str = " and ".join(doc_summary) if doc_summary else "documents"

    prompt = f"""Your name is Alfred, a helpful assistant at the University of Bristol working in the Smart Technology team.

Answer the user's question about **{target_building}** using ONLY the context below. The context includes {doc_summary_str} specific to this building.

IMPORTANT: 
- Focus your answer specifically on {target_building}
- Clearly organize information by document type (property data vs operational documentation)
- If information comes from different documents, mention which document type
- End with the document update date information

Question: {question}

Building: {target_building}

Context: {context}

{date_context}
"""

    try:
        chat = oai.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[
                {"role": "system",
                 "content": f"You are Alfred, a helpful assistant. You are answering specifically about {target_building}. Organize your response by document type (property data, operational documentation) and always include date information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        answer = chat.choices[0].message.content.strip()

        # Add metadata summary
        answer += f"\n\n**Sources for {target_building}:**\n"
        answer += f"- Property data: {len(property_data)} record(s)\n"
        answer += f"- Operational documents: {len(operational_docs)} document(s)\n"
        answer += f"- Total results: {len(target_results)}"

        return answer, publication_info
    except Exception as e:
        logging.error(f"Error generating building-focused answer: {e}")
        return enhanced_answer_with_source_date(question, top_result, all_results)


def compare_buildings_answer(question: str, results: List[Dict[str, Any]],
                             building_groups: Dict[str, List[Dict[str, Any]]]) -> Tuple[str, str]:
    """
    Generate an answer that compares information across multiple buildings.

    Args:
        question: User's question
        results: All search results
        building_groups: Results grouped by building

    Returns: (answer, publication_date_info)
    """
    # Build comparison context
    context_parts = []

    # Top 5 buildings
    for building, building_results in list(building_groups.items())[:5]:
        building_context = f"\n=== {building} ===\n"
        # Top 2 results per building
        snippets = [r.get('text', '')[:300] for r in building_results[:2]]
        building_context += "\n".join(snippets)
        context_parts.append(building_context)

    context = "\n".join(context_parts)

    buildings_list = ", ".join(list(building_groups.keys())[:5])

    prompt = f"""Your name is Alfred, a helpful assistant at the University of Bristol working in the Smart Technology team.

Answer the user's question by comparing information across multiple buildings. The context includes information from: {buildings_list}

IMPORTANT: 
- Organize your answer by building
- Highlight similarities and differences
- Be clear about which information applies to which building

Question: {question}

Context: {context}
"""

    try:
        chat = oai.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[
                {"role": "system",
                 "content": "You are Alfred, a helpful assistant. You are comparing information across multiple buildings. Organize by building and be clear about differences."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        answer = chat.choices[0].message.content.strip()
        publication_info = f"📊 Comparison across {len(building_groups)} building(s)"

        return answer, publication_info
    except Exception as e:
        logging.error(f"Error generating comparison answer: {e}")
        return "I encountered an error comparing buildings. Please try again.", ""
