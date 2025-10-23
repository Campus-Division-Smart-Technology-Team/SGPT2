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

    # Filter by building if specified
    if target_building:
        # Import here to avoid circular dependency
        try:
            from search_operations import matches_building_fuzzy
        except ImportError:
            # Fallback to original if improved version not available
            from search_operations import matches_building
            # Wrapper function with correct signature

            def matches_building_fuzzy(result: Dict[str, Any], target_building: str) -> bool:
                return matches_building(get_building_name_from_result(result), target_building)

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
                "⚠️ No operational docs found for building '%s', using top result",
                target_building
            )
            top_operational = operational_results[0]
    else:
        top_operational = operational_results[0]

    key_value = top_operational.get("key", "")
    building_name = get_building_name_from_result(top_operational)

    logging.debug(
        "Processing operational doc: %s (score: %.3f, building: %s)",
        key_value,
        top_operational.get('score', 0),
        building_name
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

    except Exception as e:  # pylint: disable=broad-except
        logging.warning(
            "Comprehensive date search failed for %s: %s", key_value, e)

    return None


def _search_metadata_date(result: Dict[str, Any]) -> Optional[str]:
    """Search metadata fields for date information."""
    metadata = result.get('metadata', {})

    for field in DATE_FIELDS_PRIORITY:
        date_value = metadata.get(field)
        if date_value:
            logging.info("Found date in metadata field '%s': %s",
                         field, date_value)
            return date_value

    return None


def get_document_dates_by_type(
    results: List[Dict[str, Any]],
    target_building: Optional[str] = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract dates from mixed Planon and operational documents.
    IMPROVED: Better building awareness.

    Args:
        results: Mixed list of search results
        target_building: Optional building to filter by

    Returns:
        (planon_date, operational_date, operational_doc_key) tuple
    """
    planon_results, operational_results = separate_results_by_type(results)

    # Get Planon date
    planon_date = find_planon_date(planon_results) if planon_results else None

    # Get operational date (building-aware)
    operational_date, operational_doc_key = find_operational_date(
        operational_results, target_building
    )

    # Log date findings
    building_name = target_building or "all buildings"
    planon_str = planon_date if planon_date else "NONE"
    operational_str = f"{operational_date} (from {operational_doc_key})" if operational_date else "NONE"

    logging.info(
        "Dates found for %s - Planon: %s, Operational: %s",
        building_name,
        planon_str,
        operational_str
    )

    return planon_date, operational_date, operational_doc_key


def format_date_information(
    planon_date: Optional[str],
    operational_date: Optional[str],
    operational_doc_key: Optional[str]
) -> Tuple[str, str]:
    """
    Format date information for LLM context and user display.

    Args:
        planon_date: Planon property condition assessment date
        operational_date: Operational/technical document date
        operational_doc_key: Key of operational document

    Returns:
        (context_str, display_str) tuple
    """
    context_parts = []
    display_parts = []

    if operational_date:
        parsed = parse_date_string(operational_date)
        display_date = format_display_date(parsed)

        context_parts.append(
            f"Technical documentation last updated: {display_date} "
            f"(source: {operational_doc_key or 'operational document'})"
        )
        display_parts.append(
            f"{EMOJI_DOCUMENT} Technical documentation last updated: **{display_date}**"
        )

    if planon_date:
        parsed = parse_date_string(planon_date)
        display_date = format_display_date(parsed)

        context_parts.append(
            f"Property condition assessment date: {display_date}"
        )
        display_parts.append(
            f"{EMOJI_BUILDING} Property assessment: **{display_date}**"
        )

    if not context_parts:
        context_str = "Date information: Not available"
        display_str = ""
    else:
        context_str = "\n".join(context_parts)
        display_str = " | ".join(display_parts)

    return context_str, display_str


def build_context_from_results(results: List[Dict[str, Any]], max_chars: int = 8000) -> str:
    """Build context string from search results with metadata."""
    context_parts = []
    total_chars = 0

    for i, r in enumerate(results, 1):
        text = get_text_from_result(r)
        doc_type = get_metadata_field(r, 'document_type', 'unknown')
        building = get_building_name_from_result(r)
        score = r.get('score', 0)

        # Add metadata header
        header = f"[Result {i} - Type: {doc_type}, Building: {building}, Score: {score:.3f}]"
        snippet = f"{header}\n{text[:1500]}"

        if total_chars + len(snippet) > max_chars:
            break

        context_parts.append(snippet)
        total_chars += len(snippet)

    return "\n\n".join(context_parts)


def build_building_grouped_context(results: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    """
    Build context grouped by document type for building-specific queries.
    IMPROVED: Uses enhanced building name extraction.
    """
    planon_results, operational_results = separate_results_by_type(results)

    context_parts = []
    total_chars = 0

    # Add operational/FRA docs first (higher priority for technical questions)
    if operational_results:
        context_parts.append("=== TECHNICAL DOCUMENTATION ===")
        for i, r in enumerate(operational_results, 1):
            text = get_text_from_result(r)
            doc_type = get_metadata_field(r, 'document_type', 'unknown')
            building = get_building_name_from_result(r)
            key = r.get('key', 'unknown')

            snippet = f"[Document {i}: {doc_type}, Building: {building}, Key: {key}]\n{text[:1200]}"

            if total_chars + len(snippet) > max_chars:
                break

            context_parts.append(snippet)
            total_chars += len(snippet)

    # Add Planon data if space remains
    if planon_results and total_chars < max_chars:
        context_parts.append("\n=== PROPERTY/PLANON DATA ===")
        for i, r in enumerate(planon_results, 1):
            text = get_text_from_result(r)
            building = get_building_name_from_result(r)

            snippet = f"[Record {i}: Building: {building}]\n{text[:800]}"

            if total_chars + len(snippet) > max_chars:
                break

            context_parts.append(snippet)
            total_chars += len(snippet)

    return "\n\n".join(context_parts)


def build_term_explanation(term_context: Optional[Dict]) -> str:
    """Build explanation of business terms for LLM context."""
    if not term_context:
        return ""

    explanations = []
    for term_key, term_info in term_context.items():
        full_name = term_info.get('full_name', term_key.upper())
        description = term_info.get('description', '')
        explanations.append(
            f"- **{term_key.upper()}** ({full_name}): {description}")

    if explanations:
        return "BUSINESS TERMS CONTEXT:\n" + "\n".join(explanations) + "\n"
    return ""


# ============================================================================
# ANSWER GENERATION FUNCTIONS
# ============================================================================


def enhanced_answer_with_source_date(
    question: str,
    top_result: Dict[str, Any],
    all_results: List[Dict[str, Any]],
    term_context: Optional[Dict] = None,
    target_building: Optional[str] = None
) -> Tuple[str, str]:
    """
    Generate answer with enhanced date awareness and building context.
    IMPROVED: Better building name extraction from results.

    Args:
        question: User query
        top_result: Top search result
        all_results: All search results
        term_context: Business term context
        target_building: Optional target building

    Returns:
        (answer, publication_info) tuple
    """
    # Separate results by type
    planon_results, operational_results = separate_results_by_type(all_results)

    # Get dates
    planon_date, operational_date, operational_doc_key = get_document_dates_by_type(
        all_results,
        target_building=target_building
    )

    date_context, publication_info = format_date_information(
        planon_date, operational_date, operational_doc_key
    )

    # Build context
    context = build_context_from_results(all_results, max_chars=8000)

    # Get building name for context
    building_name = get_building_name_from_result(top_result)
    if target_building and building_name == 'Unknown':
        building_name = target_building

    # Document summary
    doc_summary = []
    if planon_results:
        doc_summary.append(
            f"{len(planon_results)} property/Planon record(s)")
    if operational_results:
        doc_summary.append(
            f"{len(operational_results)} technical document(s)")

    doc_summary_str = " and ".join(
        doc_summary) if doc_summary else "documents"

    # Build term explanation
    term_explanation = build_term_explanation(term_context)

    prompt = f"""Your name is Alfred, a helpful assistant at the University of Bristol working in the Smart Technology team.

Answer the user's question using ONLY the context below. The context includes {doc_summary_str}.

{term_explanation}

IMPORTANT DATE INFORMATION:
- Technical documentation has a "last updated" date - use this for technical/BMS/FRA questions
- Property condition assessment date is only for building condition information
- Always prioritise and mention the technical document's last updated date for technical questions

DOCUMENT TYPES:
- Property/Planon data provides: building characteristics, conditions, facilities information
- Technical documentation provides: BMS system details, control sequences, equipment specifications, fire safety assessments, operating procedures
- When answering about technical systems, prioritise the technical documentation
- When answering about building properties/characteristics, prioritise the property data

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
