#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Answer generation using OpenAI with enhanced source date information and building-aware context.
OPTIMIZED VERSION: Works with improved search logic and multi-field building extraction.

Key improvements:
- Properly extracts building names from multiple metadata fields
- Compatible with improved fuzzy matching system
- Better handling of building-specific results
- Enhanced metadata field checking
"""
import logging
import re
from typing import Dict, List, Tuple, Any, Optional
from clients import oai
from config import ANSWER_MODEL, DEFAULT_NAMESPACE
from date_utils import (
    search_source_for_latest_date,
    parse_date_string,
    format_display_date,
    extract_date_from_single_result
)
from pinecone_utils import open_index


# ============================================================================
# CONSTANTS
# ============================================================================

DOC_TYPE_PLANON = 'planon_data'
DOC_TYPE_OPERATIONAL = 'operational_doc'
DOC_TYPE_FRA = 'fire_risk_assessment'
DOC_TYPES_TECHNICAL = [DOC_TYPE_OPERATIONAL, DOC_TYPE_FRA]

# Emojis (properly encoded)
EMOJI_DOCUMENT = "📄"
EMOJI_BUILDING = "🏢"
EMOJI_CALENDAR = "📅"
EMOJI_CHART = "📊"

# Date field priority order for metadata fallback
DATE_FIELDS_PRIORITY = ['review_date', 'updated',
                        'revised', 'date', 'document_date']

# Building name metadata fields (in priority order)
BUILDING_NAME_FIELDS = [
    'canonical_building_name',
    'building_name',
    'Property names',
    'UsrFRACondensedPropertyName'
]

# Planon date extraction patterns (compiled once)
PLANON_DATE_PATTERNS = [
    re.compile(
        r'Property condition assessment date[:\s]+([0-9]{2}\s+[A-Za-z]+\s+[0-9]{4})', re.IGNORECASE),
    re.compile(
        r'condition assessment date[:\s]+([0-9]{2}\s+[A-Za-z]+\s+[0-9]{4})', re.IGNORECASE),
    re.compile(
        r'assessment date[:\s]+([0-9]{2}\s+[A-Za-z]+\s+[0-9]{4})', re.IGNORECASE),
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_metadata_field(result: Dict[str, Any], field: str, default: Any = None) -> Any:
    """
    Get field from metadata first, then fall back to top level.
    Centralised accessor to avoid repeated pattern.
    """
    metadata = result.get('metadata', {})
    return metadata.get(field) or result.get(field, default)


def get_building_name_from_result(result: Dict[str, Any]) -> str:
    """
    Extract building name from result, checking multiple metadata fields.
    IMPROVED: Checks all building-related fields in priority order.

    Args:
        result: Search result dictionary

    Returns:
        Building name or 'Unknown' if not found
    """
    metadata = result.get('metadata', {})

    # Check each field in priority order
    for field in BUILDING_NAME_FIELDS:
        value = metadata.get(field) or result.get(field)

        if value:
            # Handle list values (take first non-empty)
            if isinstance(value, list):
                for item in value:
                    if item and str(item).strip():
                        return str(item).strip()
            # Handle string values
            elif isinstance(value, str) and value.strip():
                return value.strip()

    # Fallback: check top-level building_name (backward compatibility)
    building_name = result.get('building_name', '')
    if building_name and building_name != 'Unknown':
        return building_name

    return 'Unknown'


def get_text_from_result(result: Dict[str, Any]) -> str:
    """Extract text content from result, checking multiple locations."""
    return result.get('text', '') or get_metadata_field(result, 'text', '')


def extract_planon_date_from_text(text: str) -> Optional[str]:
    """
    Extract the property condition assessment date from Planon data text.
    Uses pre-compiled regex patterns for efficiency.
    """
    for pattern in PLANON_DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def separate_results_by_type(results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Separate results into planon and operational/FRA documents.
    Returns sorted operational results (by score descending).
    """
    planon_results = []
    operational_results = []

    for r in results:
        doc_type = get_metadata_field(r, 'document_type')

        if doc_type == DOC_TYPE_PLANON:
            planon_results.append(r)
        elif doc_type in DOC_TYPES_TECHNICAL:
            operational_results.append(r)

    # Sort operational by score (descending)
    operational_results.sort(key=lambda x: x.get('score', 0), reverse=True)

    logging.info(
        "Separated %d operational/FRA docs and %d planon records",
        len(operational_results), len(planon_results)
    )

    return planon_results, operational_results


def find_planon_date(planon_results: List[Dict[str, Any]]) -> Optional[str]:
    """Extract date from Planon property condition assessment data."""
    for result in planon_results:
        text = get_text_from_result(result)
        extracted_date = extract_planon_date_from_text(text)
        if extracted_date:
            logging.info("Found Planon assessment date: %s", extracted_date)
            return extracted_date
    return None


def find_operational_date(
    operational_results: List[Dict[str, Any]],
    target_building: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find date from operational docs, prioritising target building.
    IMPROVED: Uses enhanced building matching.

    Args:
        operational_results: List of operational/FRA documents
        target_building: Optional building to prioritise

    Returns:
        (date, doc_key) tuple
    """
    if not operational_results:
        return None, None
    # Import here to avoid circular dependency
    from search_operations import matches_building_fuzzy

    # Filter by building if specified
    if target_building:
        building_specific = [
            r for r in operational_results
            if matches_building_fuzzy(r, target_building)
        ]

        if building_specific:
            top_operational = building_specific[0]
            building_name = get_building_name_from_result(top_operational)
            logging.info(
                "✅ Using building-specific operational doc: %s (building: %s)",
                top_operational.get('key', ''),
                building_name
            )
        else:
            logging.warning(
                "No operational docs match building '%s', using top result",
                target_building
            )
            top_operational = operational_results[0]
    else:
        top_operational = operational_results[0]

    # Get date
    doc_key = get_metadata_field(top_operational, 'key')
    index_name = top_operational.get('index', '')

    if not doc_key or not index_name:
        logging.warning("Missing key or index for operational doc")
        return None, None

    try:
        idx = open_index(index_name)
        date, _ = search_source_for_latest_date(
            idx,
            doc_key,
            top_operational.get('namespace', DEFAULT_NAMESPACE)
        )
        return date, doc_key
    except Exception as e:  # pylint: disable=broad-except
        logging.error("Failed to get operational date: %s", e)
        return None, doc_key


def get_document_dates_by_type(
    results: List[Dict[str, Any]],
    target_building: Optional[str] = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract dates from different document types.

    Args:
        results: Search results
        target_building: Optional building to prioritise

    Returns:
        (planon_date, operational_date, operational_doc_key)
    """
    planon_results, operational_results = separate_results_by_type(results)

    planon_date = find_planon_date(planon_results)
    operational_date, operational_doc_key = find_operational_date(
        operational_results, target_building
    )

    return planon_date, operational_date, operational_doc_key


def format_date_information(
    planon_date: Optional[str],
    operational_date: Optional[str],
    operational_doc_key: Optional[str]
) -> Tuple[str, str]:
    """
    Format date information for prompts and display.

    Args:
        planon_date: Planon assessment date
        operational_date: Operational doc date
        operational_doc_key: Key of operational doc

    Returns:
        (date_context_for_prompt, publication_info_for_display)
    """
    date_context_parts = []
    publication_parts = []

    if operational_date:
        parsed_op = parse_date_string(operational_date)
        display_op = format_display_date(parsed_op)

        date_context_parts.append(
            f"Technical Documentation Date: {operational_date} (from document: {operational_doc_key})"
        )
        publication_parts.append(
            f"{EMOJI_DOCUMENT} Technical documentation last updated: **{display_op}**"
        )

    if planon_date:
        parsed_planon = parse_date_string(planon_date)
        display_planon = format_display_date(parsed_planon)

        date_context_parts.append(
            f"Property Condition Assessment Date: {planon_date}"
        )
        publication_parts.append(
            f"{EMOJI_BUILDING} Property condition assessed: **{display_planon}**"
        )

    if not date_context_parts:
        date_context_parts.append("Date Information: Not available")
        publication_parts.append(
            f"{EMOJI_CALENDAR} **Publication date unknown**")

    date_context = "\n".join(date_context_parts)
    publication_info = "\n".join(publication_parts)

    return date_context, publication_info


def build_context_string(results: List[Dict[str, Any]], max_chars: int = 8000) -> str:
    """
    Build context string from search results with character limit.

    Args:
        results: Search results
        max_chars: Maximum characters to include

    Returns:
        Context string
    """
    context_parts = []
    char_count = 0

    for i, result in enumerate(results, 1):
        text = get_text_from_result(result)
        doc_key = get_metadata_field(result, 'key', 'Unknown')
        doc_type = get_metadata_field(result, 'document_type', 'unknown')
        building = get_building_name_from_result(result)

        # Format result
        result_text = f"\n[Result {i}]\n"
        result_text += f"Source: {doc_key}\n"
        result_text += f"Document Type: {doc_type}\n"
        result_text += f"Building: {building}\n"
        result_text += f"Content: {text}\n"

        # Check if adding this would exceed limit
        if char_count + len(result_text) > max_chars:
            context_parts.append("\n[Additional results truncated...]")
            break

        context_parts.append(result_text)
        char_count += len(result_text)

    return "\n".join(context_parts)


def build_building_grouped_context(results: List[Dict[str, Any]], max_chars: int = 8000) -> str:
    """
    Build context grouped by building with character limit.

    Args:
        results: Search results
        max_chars: Maximum characters

    Returns:
        Context string grouped by building
    """
    # Group by building
    building_groups: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        building = get_building_name_from_result(result)
        if building not in building_groups:
            building_groups[building] = []
        building_groups[building].append(result)

    context_parts = []
    char_count = 0

    for building, building_results in building_groups.items():
        building_section = f"\n=== Building: {building} ===\n"

        for i, result in enumerate(building_results, 1):
            text = get_text_from_result(result)
            doc_key = get_metadata_field(result, 'key', 'Unknown')
            doc_type = get_metadata_field(result, 'document_type', 'unknown')

            result_text = f"\n[{building} - Result {i}]\n"
            result_text += f"Source: {doc_key}\n"
            result_text += f"Document Type: {doc_type}\n"
            result_text += f"Content: {text}\n"

            if char_count + len(building_section) + len(result_text) > max_chars:
                context_parts.append("\n[Additional results truncated...]")
                return "\n".join(context_parts)

            if i == 1:
                context_parts.append(building_section)
                char_count += len(building_section)

            context_parts.append(result_text)
            char_count += len(result_text)

    return "\n".join(context_parts)


def build_term_explanation(term_context: Optional[Dict] = None) -> str:
    """
    Build explanation of business terms detected in query.

    Args:
        term_context: Business term context

    Returns:
        Term explanation string
    """
    if not term_context:
        return ""

    explanations = []
    for term_key, term_info in term_context.items():
        explanations.append(
            f"- **{term_info['full_name']}** ({term_key.upper()}): {term_info['description']}"
        )

    if explanations:
        return "**Terms in your query:**\n" + "\n".join(explanations) + "\n"

    return ""


# ============================================================================
# MAIN ANSWER GENERATION FUNCTIONS
# ============================================================================


def enhanced_answer_with_source_date(
    question: str,
    top_result: Dict[str, Any],
    all_results: List[Dict[str, Any]],
    term_context: Optional[Dict] = None,
    target_building: Optional[str] = None
) -> Tuple[str, str]:
    """
    Generate an enhanced answer with proper date handling and building awareness.

    Args:
        question: User query
        top_result: Top search result
        all_results: All search results
        term_context: Business term context
        target_building: Optional building context

    Returns:
        (answer, publication_info) tuple
    """
    # Get dates by document type
    planon_date, operational_date, operational_doc_key = get_document_dates_by_type(
        all_results,
        target_building=target_building
    )

    # Format date information
    date_context, publication_info = format_date_information(
        planon_date, operational_date, operational_doc_key
    )

    # Build context
    context = build_context_string(all_results, max_chars=8000)

    # Build term explanation
    term_explanation = build_term_explanation(term_context)

    # Build prompt
    building_context = f"\nBuilding Context: {target_building}" if target_building else ""

    prompt = f"""Your name is Alfred, a helpful assistant at the University of Bristol working in the Smart Technology team.

Answer the user's question using ONLY the context below.

{term_explanation}

IMPORTANT DATE INFORMATION:
- Technical documentation (BMS, FRAs, operational docs) has a "last updated" date
- Property condition assessment data has an "assessment date"
- For technical questions (BMS, fire safety, systems), ALWAYS prioritise and mention the technical documentation's last updated date
- The property condition assessment date is ONLY relevant for building condition questions
- Be clear about which date applies to which type of information

DOCUMENT TYPES IN CONTEXT:
- Property/Planon data: Building characteristics, conditions, facilities
- Technical documentation: BMS systems, fire risk assessments, operating procedures
- Clearly distinguish between these when answering

Question: {question}
{building_context}

Context: {context}

{date_context}

Top Result Score: {top_result.get('score', 0):.3f}
"""

    try:
        chat = oai.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are Alfred, a helpful assistant at the University of Bristol. When answering technical questions (BMS, fire safety, systems), ALWAYS use and reference the technical document's 'last updated' date as the primary date. The property condition assessment date is ONLY for building condition information, not for technical information. Clearly distinguish between these two date types in your responses, with technical doc date taking priority for technical questions."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        content = chat.choices[0].message.content
        answer = content.strip() if content else "No answer generated."
        return answer, publication_info

    except Exception as e:  # pylint: disable=broad-except
        logging.error("Error generating answer: %s", e, exc_info=True)
        return "I encountered an error generating the answer. Please try again.", publication_info


def generate_building_focused_answer(
    question: str,
    top_result: Dict[str, Any],
    all_results: List[Dict[str, Any]],
    target_building: str,
    building_groups: Dict[str, List[Dict[str, Any]]],
    term_context: Optional[Dict] = None
) -> Tuple[str, str]:
    """
    Generate an answer specifically focused on a particular building.
    IMPROVED: Better handling of building names from metadata.

    Args:
        question: User query
        top_result: Top search result
        all_results: All search results
        target_building: Building to focus on
        building_groups: Results grouped by building
        term_context: Business term context

    Returns:
        (answer, publication_info) tuple
    """
    target_results = building_groups.get(target_building, [])

    if not target_results:
        logging.warning(
            "No results in building groups for '%s', using standard answer",
            target_building
        )
        return enhanced_answer_with_source_date(
            question, top_result, all_results, term_context, target_building
        )

    # Separate by document type
    planon_results, operational_results = separate_results_by_type(
        target_results)

    # Get dates with building context
    planon_date, operational_date, operational_doc_key = get_document_dates_by_type(
        target_results,
        target_building=target_building
    )
    date_context, publication_info = format_date_information(
        planon_date, operational_date, operational_doc_key
    )

    # Build context
    context = build_building_grouped_context(target_results, max_chars=6000)

    # Document summary
    doc_summary = []
    if planon_results:
        doc_summary.append(f"{len(planon_results)} property/Planon record(s)")
    if operational_results:
        doc_summary.append(
            f"{len(operational_results)} technical document(s)")

    doc_summary_str = " and ".join(doc_summary) if doc_summary else "documents"

    # Build prompt
    term_explanation = build_term_explanation(term_context)

    prompt = f"""Your name is Alfred, a helpful assistant at the University of Bristol working in the Smart Technology team.

Answer the user's question about **{target_building}** using ONLY the context below. The context includes {doc_summary_str} specific to this building.

{term_explanation}

IMPORTANT DATE INFORMATION:
- Technical documentation has a "last updated" date - use this for technical/BMS/FRA questions
- Property condition assessment date is only for building condition information
- Always prioritise and mention the technical document's last updated date for technical questions

DOCUMENT TYPES:
- Property/Planon data provides building characteristics, conditions, and facilities information
- Technical documentation provides BMS system details, fire risk assessments, and operating procedures
- Focus your answer specifically on {target_building}
- Clearly distinguish between property/Planon data and technical documentation

Question: {question}

Building: {target_building}

Context: {context}

{date_context}
"""

    try:
        chat = oai.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"You are Alfred, a helpful assistant. You are answering specifically about {target_building}. When answering technical questions (BMS, fire safety, systems), ALWAYS use the technical document's 'last updated' date. The property condition assessment date is ONLY for building condition information. Always distinguish between property/Planon data (building information, conditions, facilities) and technical documentation (BMS systems, controls, fire safety). Include appropriate date information based on the question type, prioritising technical doc dates for technical questions."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        content = chat.choices[0].message.content
        answer = content.strip() if content else "No answer generated."

        # Add metadata summary
        answer += f"\n\n**Information Sources for {target_building}:**"
        if planon_results:
            answer += f"\n- Property/Planon data: {len(planon_results)} record(s)"
        if operational_results:
            answer += f"\n- Technical documents: {len(operational_results)} document(s)"

        return answer, publication_info

    except Exception as e:  # pylint: disable=broad-except
        logging.error(
            "Error generating building-focused answer: %s", e, exc_info=True)
        return enhanced_answer_with_source_date(
            question, top_result, all_results, term_context, target_building
        )


def compare_buildings_answer(
    question: str,
    building_groups: Dict[str, List[Dict[str, Any]]]
) -> Tuple[str, str]:
    """
    Generate an answer that compares information across multiple buildings.
    IMPROVED: Uses enhanced building name extraction.
    """
    context_parts = []

    # Top 5 buildings
    for building, building_results in list(building_groups.items())[:5]:
        building_context = f"\n=== {building} ===\n"
        snippets = [get_text_from_result(r)[:300]
                    for r in building_results[:2]]
        building_context += "\n".join(snippets)
        context_parts.append(building_context)

    context = "\n".join(context_parts)
    buildings_list = ", ".join(list(building_groups.keys())[:5])

    prompt = f"""Your name is Alfred, a helpful assistant at the University of Bristol working in the Smart Technology team.

Answer the user's question by comparing information across multiple buildings. The context includes information from: {buildings_list}

IMPORTANT: 
- Organise your answer by building
- Highlight similarities and differences
- Be clear about which information applies to which building
- Distinguish between property data and technical documentation where relevant
- When referencing dates, be clear about whether they're technical doc dates or assessment dates

Question: {question}

Context: {context}
"""

    try:
        chat = oai.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are Alfred, a helpful assistant. You are comparing information across multiple buildings. Organise by building and be clear about differences. Distinguish between technical document dates and property assessment dates."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        content = chat.choices[0].message.content
        answer = content.strip() if content else "No answer generated."
        publication_info = f"{EMOJI_CHART} Comparison across {len(building_groups)} building(s)"

        return answer, publication_info

    except Exception as e:  # pylint: disable=broad-except
        logging.error("Error generating comparison answer: %s",
                      e, exc_info=True)
        return "I encountered an error comparing buildings. Please try again.", ""
