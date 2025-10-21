#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Answer generation using OpenAI with enhanced source date information and building-aware context.
Optimized version with building-aware date extraction and reduced duplication.
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
    Centralized accessor to avoid repeated pattern.
    """
    metadata = result.get('metadata', {})
    return metadata.get(field) or result.get(field, default)


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
    Find date from operational docs, prioritizing target building.

    Args:
        operational_results: List of operational/FRA documents
        target_building: Optional building to prioritize

    Returns:
        (date, doc_key) tuple
    """
    if not operational_results:
        return None, None

    # Filter by building if specified
    if target_building:
        # Import here to avoid circular dependency
        from search_operations import matches_building

        building_specific = [
            r for r in operational_results
            if matches_building(r.get('building_name', ''), target_building)
        ]

        if building_specific:
            top_operational = building_specific[0]
            logging.info(
                "✅ Using building-specific operational doc: %s (building: %s)",
                top_operational.get('key', ''),
                top_operational.get('building_name', '')
            )
        else:
            logging.warning(
                "⚠️ No operational docs found for building '%s', using top result",
                target_building
            )
            top_operational = operational_results[0]
    else:
        top_operational = operational_results[0]

    key_value = top_operational.get("key", "")

    logging.debug(
        "Processing operational doc: %s (score: %.3f, building: %s)",
        key_value,
        top_operational.get('score', 0),
        top_operational.get('building_name', 'Unknown')
    )

    # Strategy 1: Comprehensive search across all chunks
    operational_date = _search_comprehensive_date(top_operational, key_value)

    # Strategy 2: Check metadata fields
    if not operational_date:
        operational_date = _search_metadata_date(top_operational)

    # Strategy 3: Extract from text
    if not operational_date:
        operational_date = extract_date_from_single_result(top_operational)
        if operational_date:
            logging.info("Extracted date from text: %s", operational_date)

    if not operational_date:
        logging.warning("No date found for %s", key_value)

    return operational_date, key_value


def _search_comprehensive_date(result: Dict[str, Any], key_value: str) -> Optional[str]:
    """Search across all chunks of a document for dates."""
    if not key_value:
        return None

    idx_name = result.get("index", "")
    if not idx_name:
        return None

    try:
        idx = open_index(idx_name)
        namespace = result.get("namespace", DEFAULT_NAMESPACE)

        logging.debug(
            "Comprehensive date search: index=%s, namespace=%s, key=%s",
            idx_name, namespace, key_value
        )

        latest_date, _ = search_source_for_latest_date(
            idx, key_value, namespace)

        if latest_date:
            logging.info(
                "Found date via comprehensive search: %s", latest_date)
            return latest_date

    except (KeyError, ValueError, RuntimeError) as e:
        logging.debug("Comprehensive search failed: %s", e)

    return None


def _search_metadata_date(result: Dict[str, Any]) -> Optional[str]:
    """Search for date in metadata fields."""
    metadata = result.get('metadata', {})

    for date_field in DATE_FIELDS_PRIORITY:
        field_value = metadata.get(date_field)

        if field_value and field_value != "publication date unknown":
            logging.info(
                "Found date in metadata[%s]: %s", date_field, field_value)
            return field_value

    return None


# ============================================================================
# MAIN DATE EXTRACTION
# ============================================================================


def get_document_dates_by_type(
    results: List[Dict[str, Any]],
    target_building: Optional[str] = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Get dates by document type with building awareness.
    Always prioritizes the highest-scoring operational doc for the target building.

    Args:
        results: Search results
        target_building: Optional building to prioritize for dates

    Returns:
        (planon_date, operational_date, operational_doc_key)
    """
    planon_results, operational_results = separate_results_by_type(results)

    # Get Planon date (optionally filter by building)
    if target_building:
        # Import here to avoid circular dependency
        from search_operations import matches_building

        building_planon = [
            r for r in planon_results
            if matches_building(r.get('building_name', ''), target_building)
        ]

        if building_planon:
            planon_date = find_planon_date(building_planon)
            if planon_date:
                logging.info(
                    "Found building-specific Planon date: %s", planon_date)
        else:
            planon_date = find_planon_date(planon_results)
    else:
        planon_date = find_planon_date(planon_results)

    # Get operational date with building context
    operational_date, operational_doc_key = find_operational_date(
        operational_results,
        target_building  # Pass building context
    )

    # Summary logging
    logging.info(
        "Dates found%s - Planon: %s, Operational: %s (from %s)",
        f" for {target_building}" if target_building else "",
        planon_date or "NONE",
        operational_date or "NONE",
        operational_doc_key or "NONE"
    )

    return planon_date, operational_date, operational_doc_key


# ============================================================================
# DATE FORMATTING
# ============================================================================


def format_date_information(
    planon_date: Optional[str],
    operational_date: Optional[str],
    operational_doc_key: Optional[str] = None
) -> Tuple[str, str]:
    """
    Format date information for display and context.
    Emphasizes operational doc date as the primary "last updated" date.

    Returns:
        (date_context_for_llm, publication_info_for_display)
    """
    context_parts = []
    display_parts = []

    # Format operational document date FIRST (primary date)
    if operational_date:
        parsed = parse_date_string(operational_date)
        display_date = format_display_date(parsed)

        doc_display = _format_doc_name(
            operational_doc_key) if operational_doc_key else None

        if doc_display:
            context_parts.append(
                f"Technical documentation ('{operational_doc_key}'): Last updated {display_date}. "
                f"This is the primary date for technical/BMS/FRA information."
            )
            display_parts.append(
                f"{EMOJI_DOCUMENT} **Document last updated**: {display_date} ({doc_display})"
            )
        else:
            context_parts.append(
                f"Technical documentation: Last updated {display_date}. "
                f"This is the primary date for technical/BMS/FRA information."
            )
            display_parts.append(
                f"{EMOJI_DOCUMENT} **Document last updated**: {display_date}"
            )
    elif operational_doc_key:
        doc_display = _format_doc_name(operational_doc_key)
        context_parts.append(
            f"Technical documentation ('{operational_doc_key}'): Date not available"
        )
        display_parts.append(
            f"{EMOJI_DOCUMENT} **Document** ({doc_display}): date unknown"
        )

    # Format Planon date SECOND (supplementary)
    if planon_date:
        parsed = parse_date_string(planon_date)
        display_date = format_display_date(parsed)
        context_parts.append(
            f"Property condition assessment: {display_date}. "
            f"This date is only relevant for building condition, not BMS/technical systems."
        )
        display_parts.append(
            f"{EMOJI_BUILDING} **Property condition assessment**: {display_date}"
        )

    # Build final strings
    if context_parts:
        date_context = "Document Date Information:\n" + \
            "\n".join(f"- {p}" for p in context_parts)
    else:
        date_context = "Document Date Information: Not available"

    publication_info = "\n".join(
        display_parts) if display_parts else f"{EMOJI_CALENDAR} **Date information unavailable**"

    return date_context, publication_info


def _format_doc_name(doc_key: str) -> str:
    """Clean up document key for display."""
    return doc_key.replace('UoB-', '').replace('.pdf', '').replace('.docx', '').replace('-', ' ')


# ============================================================================
# CONTEXT BUILDING
# ============================================================================


def build_context(
    snippets: List[str],
    max_chars: int = 6000,
    prioritize_building: bool = False
) -> str:
    """
    Build context string from text snippets with optional building prioritization.
    """
    if prioritize_building:
        # Separate property data from operational docs
        property_snippets = [
            s for s in snippets if s and 'Building:' in s[:100]]
        operational_snippets = [
            s for s in snippets if s and 'Building:' not in s[:100]]

        # Combine with property data first
        snippets = property_snippets + operational_snippets

    blob = "\n\n---\n\n".join(s.strip() for s in snippets if s and s.strip())
    return blob if len(blob) <= max_chars else blob[:max_chars]


def build_building_grouped_context(
    results: List[Dict[str, Any]],
    max_chars: int = 6000
) -> str:
    """Build context grouped by building and document type."""
    building_groups = {}

    # Group by building
    for result in results:
        building = get_metadata_field(
            result, 'building_name', 'Unknown Building')
        doc_type = get_metadata_field(result, 'document_type', 'unknown')

        if building not in building_groups:
            building_groups[building] = {'planon': [], 'operational': []}

        if doc_type == DOC_TYPE_PLANON:
            building_groups[building]['planon'].append(result)
        else:
            building_groups[building]['operational'].append(result)

    # Build context with building sections
    context_parts = []
    char_count = 0

    for building, docs in building_groups.items():
        section = f"=== {building} ===\n\n"

        # Add Planon data
        if docs['planon']:
            section += "--- Property/Planon Data ---\n"
            for result in docs['planon'][:2]:
                text = get_text_from_result(result)
                section += f"{text[:500]}\n\n"

        # Add operational docs
        if docs['operational']:
            section += "--- Technical/Operational Documentation ---\n"
            for result in docs['operational'][:3]:
                text = get_text_from_result(result)
                key = get_metadata_field(result, 'key', '')
                section += f"[{key}]:\n{text[:400]}\n\n"

        if char_count + len(section) > max_chars:
            break

        context_parts.append(section)
        char_count += len(section)

        if char_count >= max_chars:
            break

    return "\n".join(context_parts)


def build_term_explanation(term_context: Optional[Dict]) -> str:
    """Build term explanation section for prompt."""
    if not term_context:
        return ""

    explanations = [
        f"- {term.upper()}: {info['full_name']} - {info['description']}"
        for term, info in term_context.items()
    ]

    return "\nRELEVANT TERMINOLOGY:\n" + "\n".join(explanations) + "\n"


# ============================================================================
# ANSWER GENERATION
# ============================================================================


def enhanced_answer_with_source_date(
    question: str,
    top_result: Dict[str, Any],
    all_results: List[Dict[str, Any]],
    term_context: Optional[Dict] = None,
    target_building: Optional[str] = None
) -> Tuple[str, str]:
    """
    Generate an answer with building-aware date extraction.

    Args:
        question: User query
        top_result: Top search result
        all_results: All search results
        term_context: Business term context
        target_building: Optional building for context

    Returns:
        (answer, publication_info) tuple
    """
    building_name = get_metadata_field(top_result, "building_name", "Unknown")

    # Use target_building if provided, otherwise use detected
    building_for_date = target_building or building_name

    # Get dates by document type WITH building context
    planon_date, operational_date, operational_doc_key = get_document_dates_by_type(
        all_results,
        target_building=building_for_date  # Pass building context
    )

    # Format date information
    date_context, publication_info = format_date_information(
        planon_date, operational_date, operational_doc_key
    )

    # Build context
    snippets = [get_text_from_result(r)
                for r in all_results if get_text_from_result(r)]
    context = build_context(snippets, prioritize_building=True)

    # Build prompt
    term_explanation = build_term_explanation(term_context)

    prompt = f"""Your name is Alfred, a helpful assistant at the University of Bristol working in the Smart Technology team.

Answer the user's question using ONLY the context below. If the answer cannot be found in the context, tell the user that you don't know.

{term_explanation}

IMPORTANT DATE INFORMATION:
- Technical documentation (BMS/operational docs and Fire Risk Assessments) has a "last updated" date - this is when the documentation was last revised
- Property/Planon data includes a property condition assessment date - this is when the building condition was assessed
- For technical questions (BMS, fire safety, systems), ALWAYS reference the technical document's last updated date
- The property condition assessment date is only relevant when discussing building condition or facilities

DOCUMENT TYPES:
- Property/Planon data provides: building characteristics, location, size, facilities manager, condition ratings
- Technical documentation provides: BMS system details, control sequences, equipment specifications, fire safety assessments, operating procedures
- When answering about technical systems, prioritize the technical documentation
- When answering about building properties/characteristics, prioritize the property data

Question: {question}

Context: {context}

{date_context}

Building: {building_name}
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

    except Exception as e:
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
        doc_summary.append(f"{len(operational_results)} technical document(s)")

    doc_summary_str = " and ".join(doc_summary) if doc_summary else "documents"

    # Build prompt
    term_explanation = build_term_explanation(term_context)

    prompt = f"""Your name is Alfred, a helpful assistant at the University of Bristol working in the Smart Technology team.

Answer the user's question about **{target_building}** using ONLY the context below. The context includes {doc_summary_str} specific to this building.

{term_explanation}

IMPORTANT DATE INFORMATION:
- Technical documentation has a "last updated" date - use this for technical/BMS/FRA questions
- Property condition assessment date is only for building condition information
- Always prioritize and mention the technical document's last updated date for technical questions

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
                    "content": f"You are Alfred, a helpful assistant. You are answering specifically about {target_building}. When answering technical questions (BMS, fire safety, systems), ALWAYS use the technical document's 'last updated' date. The property condition assessment date is ONLY for building condition information. Always distinguish between property/Planon data (building information, conditions, facilities) and technical documentation (BMS systems, controls, fire safety). Include appropriate date information based on the question type, prioritizing technical doc dates for technical questions."
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

    except Exception as e:
        logging.error(
            "Error generating building-focused answer: %s", e, exc_info=True)
        return enhanced_answer_with_source_date(
            question, top_result, all_results, term_context, target_building
        )


def compare_buildings_answer(
    question: str,
    building_groups: Dict[str, List[Dict[str, Any]]]
) -> Tuple[str, str]:
    """Generate an answer that compares information across multiple buildings."""
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
- Organize your answer by building
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
                    "content": "You are Alfred, a helpful assistant. You are comparing information across multiple buildings. Organize by building and be clear about differences. Distinguish between technical document dates and property assessment dates."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        content = chat.choices[0].message.content
        answer = content.strip() if content else "No answer generated."
        publication_info = f"{EMOJI_CHART} Comparison across {len(building_groups)} building(s)"

        return answer, publication_info

    except Exception as e:
        logging.error("Error generating comparison answer: %s",
                      e, exc_info=True)
        return "I encountered an error comparing buildings. Please try again.", ""
