#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Answer generation using OpenAI with enhanced source date information and building-aware context.
"""

from typing import Dict, List, Tuple, Any, Optional
from clients import oai
from config import ANSWER_MODEL, DEFAULT_NAMESPACE
from date_utils import search_source_for_latest_date, parse_date_string, format_display_date
from pinecone_utils import open_index
import logging
import re


def extract_planon_date_from_text(text: str) -> Optional[str]:
    """
    Extract the property condition assessment date from Planon data text.
    Looks for patterns like "Property condition assessment date: 03 November 2021"
    """
    patterns = [
        r'Property condition assessment date[:\s]+([0-9]{2}\s+[A-Za-z]+\s+[0-9]{4})',
        r'condition assessment date[:\s]+([0-9]{2}\s+[A-Za-z]+\s+[0-9]{4})',
        r'assessment date[:\s]+([0-9]{2}\s+[A-Za-z]+\s+[0-9]{4})',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def get_document_dates_by_type(results: List[Dict[str, Any]],
                               target_building: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Get separate dates for Planon data and operational documents.

    Returns:
        (planon_date, operational_date, operational_doc_key)
    """
    planon_date = None
    operational_date = None
    operational_doc_key = None

    # Separate results by type
    planon_results = [r for r in results if r.get(
        'document_type') == 'planon_data']
    operational_results = [r for r in results if r.get(
        'document_type') == 'operational_doc']

    # Get Planon date from property condition assessment
    if planon_results:
        for result in planon_results:
            text = result.get('text', '')
            extracted_date = extract_planon_date_from_text(text)
            if extracted_date:
                planon_date = extracted_date
                logging.info(f"Found Planon assessment date: {planon_date}")
                break

        # Fallback to last_modified if available
        if not planon_date:
            for result in planon_results:
                last_mod = result.get('last_modified')
                if last_mod:
                    planon_date = last_mod
                    break

    # Get operational document date - ALWAYS from operational docs, never from Planon
    if operational_results:
        # Get the highest-scoring operational document
        top_operational = operational_results[0]
        key_value = top_operational.get("key", "")
        operational_doc_key = key_value
        idx_name = top_operational.get("index", "")

        if key_value and idx_name:
            try:
                idx = open_index(idx_name)
                latest_date, _ = search_source_for_latest_date(
                    idx, key_value, top_operational.get(
                        "namespace", DEFAULT_NAMESPACE)
                )
                if latest_date:
                    operational_date = latest_date
                    logging.info(
                        f"Found operational doc date from {key_value}: {operational_date}")
            except Exception as e:
                logging.error(f"Error fetching operational date: {e}")

        # Fallback to last_modified from operational doc
        if not operational_date:
            operational_date = top_operational.get('last_modified')
            if operational_date:
                logging.info(
                    f"Using last_modified from operational doc: {operational_date}")

    return planon_date, operational_date, operational_doc_key


def format_date_information(planon_date: Optional[str],
                            operational_date: Optional[str],
                            operational_doc_key: Optional[str] = None) -> Tuple[str, str]:
    """
    Format date information for display and context.

    Args:
        planon_date: Property condition assessment date
        operational_date: Operational document date
        operational_doc_key: Key/filename of operational document

    Returns:
        (date_context_for_llm, publication_info_for_display)
    """
    context_parts = []
    display_parts = []

    # Format Planon date
    if planon_date:
        parsed = parse_date_string(planon_date)
        display_date = format_display_date(parsed)
        context_parts.append(
            f"Property/Planon data (building characteristics, facilities, condition): Property condition assessment date is {display_date}")
        display_parts.append(
            f"📊 **Planon property data**: as of **{display_date}** (property condition assessment)")

    # Format operational document date
    if operational_date and operational_doc_key:
        parsed = parse_date_string(operational_date)
        display_date = format_display_date(parsed)

        # Clean up the document key for display (remove path and .pdf)
        doc_display = operational_doc_key.replace(
            'UoB-', '').replace('.pdf', '').replace('-', ' ')

        context_parts.append(
            f"BMS/Operational documentation (technical systems, controls, procedures): Document '{operational_doc_key}' last updated {display_date}")
        display_parts.append(
            f"📄 **BMS/Operational documentation** ({doc_display}): last updated **{display_date}**")
    elif operational_doc_key:
        # We have the doc but no date
        doc_display = operational_doc_key.replace(
            'UoB-', '').replace('.pdf', '').replace('-', ' ')
        context_parts.append(
            f"BMS/Operational documentation (technical systems, controls, procedures): Document '{operational_doc_key}' (date not available)")
        display_parts.append(
            f"📄 **BMS/Operational documentation** ({doc_display}): date unknown")

    # Build the final strings
    if context_parts:
        date_context = "Document Date Information:\n" + \
            "\n".join(f"- {part}" for part in context_parts)
    else:
        date_context = "Document Date Information: Not available"

    publication_info = "\n".join(
        display_parts) if display_parts else "📅 **Date information unavailable**"

    return date_context, publication_info


def build_context(snippets: List[str], max_chars: int = 6000,
                  prioritize_building: bool = False) -> str:
    """
    Build context string from text snippets with optional building prioritization.
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
    Build context grouped by building and document type for better organization.
    """
    building_groups = {}

    # Group by building
    for result in results:
        building = result.get('building_name', 'Unknown Building')
        if building not in building_groups:
            building_groups[building] = {'planon': [], 'operational': []}

        doc_type = result.get('document_type', 'unknown')
        if doc_type == 'planon_data':
            building_groups[building]['planon'].append(result)
        else:
            building_groups[building]['operational'].append(result)

    # Build context with building sections
    context_parts = []
    char_count = 0

    for building, docs in building_groups.items():
        section = f"=== {building} ===\n\n"

        # Add Planon data first
        if docs['planon']:
            section += "--- Property/Planon Data ---\n"
            for result in docs['planon'][:2]:  # Limit to 2 property records
                text = result.get('text', '')
                section += f"{text[:500]}\n\n"

        # Add operational docs
        if docs['operational']:
            section += "--- BMS/Operational Documentation ---\n"
            for result in docs['operational'][:3]:  # Limit to 3 operational docs
                text = result.get('text', '')
                key = result.get('key', '')
                section += f"[{key}]:\n{text[:400]}\n\n"

        if char_count + len(section) > max_chars:
            break

        context_parts.append(section)
        char_count += len(section)

        if char_count >= max_chars:
            break

    return "\n".join(context_parts)


def enhanced_answer_with_source_date(question: str, top_result: Dict[str, Any],
                                     all_results: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Generate an answer that includes separate date information for Planon and operational docs.
    Always uses operational document for the "document last updated" even if Planon ranks higher.

    Returns: (answer, publication_date_info)
    """
    building_name = top_result.get("building_name", "Unknown")

    # Get dates by document type
    planon_date, operational_date, operational_doc_key = get_document_dates_by_type(
        all_results, building_name)

    # Format date information
    date_context, publication_info = format_date_information(
        planon_date, operational_date, operational_doc_key
    )

    # Build context with building prioritization
    snippets = [r.get("text", "") for r in all_results if r.get("text")]
    context = build_context(snippets, prioritize_building=True)

    # Determine what type of question this is
    question_lower = question.lower()
    is_bms_query = any(term in question_lower for term in [
                       'bms', 'building management', 'controls', 'hvac', 'system', 'frost', 'temperature', 'monitoring'])
    is_property_query = any(term in question_lower for term in [
                            'area', 'size', 'condition', 'rating', 'manager', 'facilities', 'campus', 'location'])

    prompt = f"""Your name is Alfred, a helpful assistant at the University of Bristol working in the Smart Technology team.

Answer the user's question using ONLY the context below. If the answer cannot be found in the context, tell the user that Regan has told you to say you don't know.

IMPORTANT: 
- The context includes both property/Planon data AND BMS/operational documentation
- Property/Planon data provides: building characteristics, location, size, facilities manager, condition ratings, fire ratings
- BMS/operational documentation provides: technical system details, control sequences, equipment specifications, operating procedures
- When answering about BMS/technical systems, prioritize the operational documentation
- When answering about building properties/characteristics, prioritize the property data
- Always end your response with information about the relevant document dates
- The property condition assessment date tells when the building condition was last assessed
- The BMS/operational document date tells when the technical documentation was last updated

Question: {question}

Context: {context}

{date_context}

Building: {building_name}
Top Result Score: {top_result.get('score', 'Unknown'):.3f}
"""

    try:
        chat = oai.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[
                {"role": "system",
                 "content": "You are Alfred, a helpful assistant at the University of Bristol. Always distinguish between property/Planon data (building characteristics) and BMS/operational documentation (technical systems). Include appropriate date information based on what the user is asking about. For BMS questions, emphasize the operational documentation date. For property questions, emphasize the assessment date."},
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
    Generate an answer specifically focused on a particular building with separate date tracking.
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

    # Get dates by type
    planon_date, operational_date, operational_doc_key = get_document_dates_by_type(
        target_results, target_building)

    # Format date information
    date_context, publication_info = format_date_information(
        planon_date, operational_date, operational_doc_key
    )

    # Build grouped context
    context = build_building_grouped_context(target_results, max_chars=6000)

    # Create document type summary
    doc_summary = []
    if property_data:
        doc_summary.append(f"{len(property_data)} property/Planon record(s)")
    if operational_docs:
        doc_summary.append(
            f"{len(operational_docs)} BMS/operational document(s)")

    doc_summary_str = " and ".join(doc_summary) if doc_summary else "documents"

    prompt = f"""Your name is Alfred, a helpful assistant at the University of Bristol working in the Smart Technology team.

Answer the user's question about **{target_building}** using ONLY the context below. The context includes {doc_summary_str} specific to this building.

IMPORTANT: 
- Focus your answer specifically on {target_building}
- Clearly distinguish between property/Planon data and BMS/operational documentation
- Property/Planon data provides building characteristics, conditions, and facilities information
- BMS/operational documentation provides technical system details and operating procedures
- Always mention the relevant dates for the information you're providing
- End with clear date information

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
                 "content": f"You are Alfred, a helpful assistant. You are answering specifically about {target_building}. Always distinguish between property/Planon data (building information, conditions, facilities) and BMS/operational documentation (technical systems, controls). Include appropriate date information based on the question type."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        answer = chat.choices[0].message.content.strip()

        # Add metadata summary
        answer += f"\n\n**Information Sources for {target_building}:**"
        if property_data:
            answer += f"\n- Property/Planon data: {len(property_data)} record(s)"
        if operational_docs:
            answer += f"\n- BMS/Operational documents: {len(operational_docs)} document(s)"

        return answer, publication_info
    except Exception as e:
        logging.error(f"Error generating building-focused answer: {e}")
        return enhanced_answer_with_source_date(question, top_result, all_results)


def compare_buildings_answer(question: str, results: List[Dict[str, Any]],
                             building_groups: Dict[str, List[Dict[str, Any]]]) -> Tuple[str, str]:
    """
    Generate an answer that compares information across multiple buildings.
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
- Distinguish between property data and BMS/operational data where relevant

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
