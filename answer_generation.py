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
    ALWAYS prioritizes the highest-scoring operational_doc for the "last updated" date.

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

    # Sort operational results by score to get the highest-scoring one
    operational_results = sorted(
        operational_results, key=lambda x: x.get('score', 0), reverse=True)

    logging.info(
        f"Found {len(operational_results)} operational docs and {len(planon_results)} planon records")

    # Get Planon date ONLY from property condition assessment
    if planon_results:
        for result in planon_results:
            text = result.get('text', '')
            extracted_date = extract_planon_date_from_text(text)
            if extracted_date:
                planon_date = extracted_date
                logging.info(f"✓ Found Planon assessment date: {planon_date}")
                break

    # Get operational document date from HIGHEST-SCORING operational doc
    if operational_results:
        # Get the highest-scoring operational document
        top_operational = operational_results[0]
        key_value = top_operational.get("key", "")
        operational_doc_key = key_value
        idx_name = top_operational.get("index", "")

        logging.info(
            f"Using highest-scoring operational doc: {key_value} (score: {top_operational.get('score', 0):.3f})")

        if key_value and idx_name:
            try:
                idx = open_index(idx_name)
                namespace = top_operational.get("namespace", DEFAULT_NAMESPACE)
                logging.info(
                    f"Searching for dates in index={idx_name}, namespace={namespace}, key={key_value}")

                latest_date, _ = search_source_for_latest_date(
                    idx, key_value, namespace
                )
                if latest_date:
                    operational_date = latest_date
                    logging.info(
                        f"✓ Found operational doc date from {key_value}: {operational_date}")
                else:
                    logging.warning(
                        f"✗ No date found in search_source_for_latest_date for {key_value}")
            except Exception as e:
                logging.error(
                    f"✗ Error fetching operational date: {e}", exc_info=True)

        # Fallback to metadata if full search didn't work
        if not operational_date:
            # Try last_modified from metadata
            operational_date = top_operational.get('last_modified')
            if operational_date:
                logging.info(
                    f"Using last_modified from operational doc metadata: {operational_date}")
            else:
                # Try to extract from text as last resort
                text = top_operational.get('text', '')
                from date_utils import extract_date_from_single_result
                operational_date = extract_date_from_single_result(
                    top_operational)
                if operational_date:
                    logging.info(
                        f"Extracted date from operational doc text: {operational_date}")

    return planon_date, operational_date, operational_doc_key


def format_date_information(planon_date: Optional[str],
                            operational_date: Optional[str],
                            operational_doc_key: Optional[str] = None) -> Tuple[str, str]:
    """
    Format date information for display and context.
    Emphasizes operational doc date as the primary "last updated" date.

    Args:
        planon_date: Property condition assessment date
        operational_date: Operational document last updated date
        operational_doc_key: Key/filename of operational document

    Returns:
        (date_context_for_llm, publication_info_for_display)
    """
    context_parts = []
    display_parts = []

    # Format operational document date FIRST (it's the primary date)
    if operational_date and operational_doc_key:
        parsed = parse_date_string(operational_date)
        display_date = format_display_date(parsed)

        # Clean up the document key for display (remove path and .pdf)
        doc_display = operational_doc_key.replace(
            'UoB-', '').replace('.pdf', '').replace('-', ' ')

        context_parts.append(
            f"BMS/Operational documentation ('{operational_doc_key}'): Last updated {display_date}. This is the primary date for technical/BMS information.")
        display_parts.append(
            f"📄 **Document last updated**: **{display_date}** ({doc_display})")
    elif operational_doc_key:
        # We have the doc but no date
        doc_display = operational_doc_key.replace(
            'UoB-', '').replace('.pdf', '').replace('-', ' ')
        context_parts.append(
            f"BMS/Operational documentation ('{operational_doc_key}'): Date not available")
        display_parts.append(
            f"📄 **Document** ({doc_display}): date unknown")

    # Format Planon date SECOND (supplementary information)
    if planon_date:
        parsed = parse_date_string(planon_date)
        display_date = format_display_date(parsed)
        context_parts.append(
            f"Property condition assessment: {display_date}. This date is only relevant when discussing building condition, not BMS/technical systems.")
        display_parts.append(
            f"🏢 **Property condition assessment**: {display_date}")

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
    ALWAYS uses the highest-scoring operational document for the "document last updated" date.

    Returns: (answer, publication_date_info)
    """
    building_name = top_result.get("building_name", "Unknown")

    # Get dates by document type - prioritizing highest-scoring operational doc
    planon_date, operational_date, operational_doc_key = get_document_dates_by_type(
        all_results, building_name)

    # Format date information with operational doc date first
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

IMPORTANT DATE INFORMATION:
- BMS/Operational documentation has a "last updated" date - this is when the technical documentation was last revised
- Property/Planon data includes a property condition assessment date - this is when the building condition was assessed
- For BMS/technical questions, ALWAYS reference the operational document's last updated date
- The property condition assessment date is only relevant when discussing building condition or facilities

DOCUMENT TYPES:
- Property/Planon data provides: building characteristics, location, size, facilities manager, condition ratings, fire ratings
- BMS/operational documentation provides: technical system details, control sequences, equipment specifications, operating procedures
- When answering about BMS/technical systems, prioritize the operational documentation
- When answering about building properties/characteristics, prioritize the property data

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
                 "content": "You are Alfred, a helpful assistant at the University of Bristol. When answering BMS/technical questions, ALWAYS use and reference the operational document's 'last updated' date as the primary date. The property condition assessment date is ONLY for building condition information, not for BMS/technical information. Clearly distinguish between these two date types in your responses, with operational doc date taking priority for technical questions."},
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
    Uses highest-scoring operational doc for "last updated" date.
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

    # Get dates by type - prioritizing highest-scoring operational doc
    planon_date, operational_date, operational_doc_key = get_document_dates_by_type(
        target_results, target_building)

    # Format date information with operational doc date first
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

IMPORTANT DATE INFORMATION:
- BMS/Operational documentation has a "last updated" date - use this for technical/BMS questions
- Property condition assessment date is only for building condition information
- Always prioritize and mention the operational document's last updated date for BMS questions

DOCUMENT TYPES:
- Property/Planon data provides building characteristics, conditions, and facilities information
- BMS/operational documentation provides technical system details and operating procedures
- Focus your answer specifically on {target_building}
- Clearly distinguish between property/Planon data and BMS/operational documentation

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
                 "content": f"You are Alfred, a helpful assistant. You are answering specifically about {target_building}. When answering BMS/technical questions, ALWAYS use the operational document's 'last updated' date. The property condition assessment date is ONLY for building condition information. Always distinguish between property/Planon data (building information, conditions, facilities) and BMS/operational documentation (technical systems, controls). Include appropriate date information based on the question type, prioritizing operational doc dates for technical questions."},
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
- When referencing dates, be clear about whether they're operational doc dates or assessment dates

Question: {question}

Context: {context}
"""

    try:
        chat = oai.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[
                {"role": "system",
                 "content": "You are Alfred, a helpful assistant. You are comparing information across multiple buildings. Organize by building and be clear about differences. Distinguish between operational document dates and property assessment dates."},
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
