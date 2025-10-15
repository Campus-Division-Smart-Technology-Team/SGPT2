#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Answer generation using OpenAI with enhanced source date information and building-aware context.
"""
import logging
import re
from typing import Dict, List, Tuple, Any, Optional
from clients import oai
from config import ANSWER_MODEL, DEFAULT_NAMESPACE
from date_utils import search_source_for_latest_date, parse_date_string, format_display_date, extract_date_from_single_result
from pinecone_utils import open_index


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


def get_document_dates_by_type(results: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Get separate dates for Planon data and operational/FRA documents.
    ALWAYS prioritises the highest-scoring operational_doc or fire_risk_assessment for the "last updated" date.

    Returns:
        (planon_date, operational_date, operational_doc_key)
    """
    planon_date = None
    operational_date = None
    operational_doc_key = None

    # Separate results by type - CHECK METADATA for document_type
    planon_results = []
    operational_results = []

    for r in results:
        # Get document_type from metadata, not top level
        metadata = r.get('metadata', {})
        doc_type = metadata.get('document_type') or r.get('document_type')

        if doc_type == 'planon_data':
            planon_results.append(r)
        elif doc_type in ['operational_doc', 'fire_risk_assessment']:
            operational_results.append(r)

    # Sort operational results by score to get the highest-scoring one
    operational_results = sorted(
        operational_results, key=lambda x: x.get('score', 0), reverse=True)

    logging.info(
        "Found %d operational/FRA docs and %d planon records",
        len(operational_results), len(planon_results))

    # Get Planon date ONLY from property condition assessment
    if planon_results:
        for result in planon_results:
            text = result.get('text', '') or result.get(
                'metadata', {}).get('text', '')
            extracted_date = extract_planon_date_from_text(text)
            if extracted_date:
                planon_date = extracted_date
                logging.info(
                    "[SUCCESS] Found Planon assessment date: %s", planon_date)
                break

    # Get operational document date from HIGHEST-SCORING operational or FRA doc
    if operational_results:
        # Get the highest-scoring operational/FRA document
        top_operational = operational_results[0]
        key_value = top_operational.get("key", "")
        operational_doc_key = key_value
        metadata = top_operational.get('metadata', {})

        logging.info(
            "Processing highest-scoring operational/FRA doc: %s (score: %.3f)",
            key_value, top_operational.get('score', 0))

        # DEBUG: Log full metadata structure
        logging.info("Available metadata fields: %s", list(metadata.keys()))
        logging.info("Document key: %s", key_value)

        # CRITICAL: Use search_source_for_latest_date to properly find the document date
        # by searching through ALL chunks of this document
        if key_value:
            idx_name = top_operational.get("index", "")
            if idx_name:
                try:
                    idx = open_index(idx_name)
                    namespace = top_operational.get(
                        "namespace", DEFAULT_NAMESPACE)
                    logging.info(
                        "Searching for dates across all chunks in index=%s, namespace=%s, key=%s",
                        idx_name, namespace, key_value)

                    latest_date, _ = search_source_for_latest_date(
                        idx, key_value, namespace
                    )
                    if latest_date:
                        operational_date = latest_date
                        logging.info(
                            "[SUCCESS] Found date via comprehensive search: %s", operational_date)
                    else:
                        logging.warning(
                            "[FAILED] No date found in comprehensive search for %s", key_value)
                except (KeyError, ValueError, RuntimeError) as e:
                    logging.error(
                        "[FAILED] Error fetching operational date: %s", e, exc_info=True)
            else:
                logging.warning(
                    "[FAILED] No index name available for date search")

        # FALLBACK: Try metadata fields (but skip last_modified as it's not reliable)
        if not operational_date:
            logging.info("Comprehensive search failed, checking metadata...")
            for date_field in ['review_date', 'updated', 'revised', 'date', 'document_date']:
                field_value = metadata.get(date_field)
                logging.info(
                    "Checking metadata[%s]: %s", date_field, field_value)

                if field_value and field_value != "publication date unknown":
                    operational_date = field_value
                    logging.info(
                        "[SUCCESS] Found date in metadata[%s]: %s", date_field, operational_date)
                    break

        # LAST RESORT: Extract from text
        if not operational_date:
            logging.info("Metadata search failed, trying text extraction...")
            operational_date = extract_date_from_single_result(top_operational)
            if operational_date:
                logging.info(
                    "[SUCCESS] Extracted date from operational doc text: %s", operational_date)
            else:
                logging.warning(
                    "[FAILED] No date found for operational doc %s using any method", key_value)

    # Final summary log
    logging.info("=" * 60)
    logging.info("DATE EXTRACTION SUMMARY:")
    logging.info("Planon date: %s", planon_date or "NOT FOUND")
    logging.info("Operational date: %s", operational_date or "NOT FOUND")
    logging.info("Operational doc key: %s", operational_doc_key or "NOT FOUND")
    logging.info("=" * 60)

    return planon_date, operational_date, operational_doc_key


def format_date_information(planon_date: Optional[str],
                            operational_date: Optional[str],
                            operational_doc_key: Optional[str] = None) -> Tuple[str, str]:
    """
    Format date information for display and context.
    Emphasises operational doc date as the primary "last updated" date.

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
    if operational_date:
        parsed = parse_date_string(operational_date)
        display_date = format_display_date(parsed)

        if operational_doc_key:
            doc_display = operational_doc_key.replace(
                'UoB-', '').replace('.pdf', '').replace('.docx', '').replace('-', ' ')

            context_parts.append(
                f"Technical documentation ('{operational_doc_key}'): Last updated {display_date}. This is the primary date for technical/BMS/FRA information.")
            display_parts.append(
                f"📄 **Document last updated**: {display_date} ({doc_display})")
        else:
            context_parts.append(
                f"Technical documentation: Last updated {display_date}. This is the primary date for technical/BMS/FRA information.")
            display_parts.append(
                f"📄 **Document last updated**: {display_date}")
    elif operational_doc_key:
        doc_display = operational_doc_key.replace(
            'UoB-', '').replace('.pdf', '').replace('.docx', '').replace('-', ' ')
        context_parts.append(
            f"Technical documentation ('{operational_doc_key}'): Date not available")
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
                  prioritise_building: bool = False) -> str:
    """
    Build context string from text snippets with optional building prioritization.
    """
    if prioritise_building:
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
        # Get building_name from metadata first, then fallback to top level
        metadata = result.get('metadata', {})
        building = metadata.get('building_name') or result.get(
            'building_name', 'Unknown Building')

        if building not in building_groups:
            building_groups[building] = {'planon': [], 'operational': []}

        # Get document_type from metadata first, then fallback to top level
        doc_type = metadata.get('document_type') or result.get(
            'document_type', 'unknown')

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
                text = result.get('text', '') or result.get(
                    'metadata', {}).get('text', '')
                section += f"{text[:500]}\n\n"

        # Add operational docs
        if docs['operational']:
            section += "--- Technical/Operational Documentation ---\n"
            for result in docs['operational'][:3]:  # Limit to 3 operational docs
                text = result.get('text', '') or result.get(
                    'metadata', {}).get('text', '')
                key = result.get('key', '') or result.get(
                    'metadata', {}).get('key', '')
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
    """
    # DEBUG: Log top result structure
    logging.info("=" * 60)
    logging.info("TOP RESULT DEBUG:")
    logging.info("Key: %s", top_result.get('key'))
    metadata = top_result.get('metadata', {})
    logging.info("Document type: %s", metadata.get(
        'document_type') or top_result.get('document_type'))
    logging.info("Building: %s", metadata.get('building_name')
                 or top_result.get('building_name'))
    logging.info("Metadata keys: %s", list(metadata.keys()))
    logging.info("=" * 60)

    # Get building name from metadata first, then fallback to top level
    building_name = metadata.get('building_name') or top_result.get(
        "building_name", "Unknown")

    # Get dates by document type - NOW RETURNS 3 VALUES
    planon_date, operational_date, operational_doc_key = get_document_dates_by_type(
        all_results)

    # Format date information with operational doc date first
    date_context, publication_info = format_date_information(
        planon_date, operational_date, operational_doc_key
    )

    # Build context with building prioritisation
    snippets = []
    for r in all_results:
        text = r.get("text", "") or r.get('metadata', {}).get('text', "")
        if text:
            snippets.append(text)

    context = build_context(snippets, prioritise_building=True)

    prompt = f"""Your name is Alfred, a helpful assistant at the University of Bristol working in the Smart Technology team.

Answer the user's question using ONLY the context below. If the answer cannot be found in the context, tell the user that you don't know.

IMPORTANT DATE INFORMATION:
- Technical documentation (BMS/operational docs and Fire Risk Assessments) has a "last updated" date - this is when the documentation was last revised
- Property/Planon data includes a property condition assessment date - this is when the building condition was assessed
- For technical questions (BMS, fire safety, systems), ALWAYS reference the technical document's last updated date
- The property condition assessment date is only relevant when discussing building condition or facilities

DOCUMENT TYPES:
- Property/Planon data provides: building characteristics, location, size, facilities manager, condition ratings
- Technical documentation provides: BMS system details, control sequences, equipment specifications, fire safety assessments, operating procedures
- When answering about technical systems, prioritise the technical documentation
- When answering about building properties/characteristics, prioritise the property data

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
                 "content": "You are Alfred, a helpful assistant at the University of Bristol. When answering technical questions (BMS, fire safety, systems), ALWAYS use and reference the technical document's 'last updated' date as the primary date. The property condition assessment date is ONLY for building condition information, not for technical information. Clearly distinguish between these two date types in your responses, with technical doc date taking priority for technical questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        content = chat.choices[0].message.content
        answer = content.strip() if content is not None else "No answer generated."
        return answer, publication_info
    except Exception as e:
        logging.error("Error generating answer: %s", e)
        return "I encountered an error generating the answer. Please try again.", publication_info


def generate_building_focused_answer(question: str, top_result: Dict[str, Any],
                                     all_results: List[Dict[str, Any]],
                                     target_building: str,
                                     building_groups: Dict[str, List[Dict[str, Any]]]) -> Tuple[str, str]:
    """
    Generate an answer specifically focused on a particular building with separate date tracking.
    Uses highest-scoring operational_doc or fire_risk_assessment for "last updated" date.
    """
    # Get results specific to target building
    target_results = building_groups.get(target_building, [])

    if not target_results:
        # Fallback to standard answer generation
        return enhanced_answer_with_source_date(question, top_result, all_results)

    # Separate by document type - check metadata
    property_data = []
    operational_docs = []

    for r in target_results:
        metadata = r.get('metadata', {})
        doc_type = metadata.get('document_type') or r.get('document_type')

        if doc_type == 'planon_data':
            property_data.append(r)
        elif doc_type in ['operational_doc', 'fire_risk_assessment']:
            operational_docs.append(r)

    # Get dates by type - NOW RETURNS 3 VALUES
    planon_date, operational_date, operational_doc_key = get_document_dates_by_type(
        target_results)

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
            f"{len(operational_docs)} technical document(s)")

    doc_summary_str = " and ".join(doc_summary) if doc_summary else "documents"

    prompt = f"""Your name is Alfred, a helpful assistant at the University of Bristol working in the Smart Technology team.

Answer the user's question about **{target_building}** using ONLY the context below. The context includes {doc_summary_str} specific to this building.

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
                {"role": "system",
                 "content": f"You are Alfred, a helpful assistant. You are answering specifically about {target_building}. When answering technical questions (BMS, fire safety, systems), ALWAYS use the technical document's 'last updated' date. The property condition assessment date is ONLY for building condition information. Always distinguish between property/Planon data (building information, conditions, facilities) and technical documentation (BMS systems, controls, fire safety). Include appropriate date information based on the question type, prioritizing technical doc dates for technical questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        content = chat.choices[0].message.content
        answer = content.strip() if content is not None else "No answer generated."

        # Add metadata summary
        answer += f"\n\n**Information Sources for {target_building}:**"
        if property_data:
            answer += f"\n- Property/Planon data: {len(property_data)} record(s)"
        if operational_docs:
            answer += f"\n- Technical documents: {len(operational_docs)} document(s)"

        return answer, publication_info
    except Exception as e:
        logging.error("Error generating building-focused answer: %s", e)
        return enhanced_answer_with_source_date(question, top_result, all_results)


def compare_buildings_answer(question: str,
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
        snippets = []
        for r in building_results[:2]:
            text = r.get('text', '') or r.get('metadata', {}).get('text', '')
            snippets.append(text[:300])
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
                {"role": "system",
                 "content": "You are Alfred, a helpful assistant. You are comparing information across multiple buildings. Organise by building and be clear about differences. Distinguish between technical document dates and property assessment dates."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        content = chat.choices[0].message.content
        answer = content.strip() if content is not None else "No answer generated."
        publication_info = f"📊 Comparison across {len(building_groups)} building(s)"

        return answer, publication_info
    except Exception as e:
        logging.error("Error generating comparison answer: %s", e)
        return "I encountered an error comparing buildings. Please try again.", ""
