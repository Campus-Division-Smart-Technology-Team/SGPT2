#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Counting and aggregation queries for AskAlfred.

This module handles queries that require counting or aggregating across
all documents, which the standard RAG search cannot handle effectively.
"""

import logging
from typing import Dict, List, Any, Optional, Set
import re

from pinecone_utils import open_index, normalise_matches
from config import TARGET_INDEXES, DEFAULT_NAMESPACE, get_index_config
from building_utils import BUILDING_METADATA_FIELDS


# ============================================================================
# QUERY DETECTION
# ============================================================================

# Patterns to detect counting queries
COUNTING_PATTERNS = [
    re.compile(r'\bhow\s+many\s+buildings?\b', re.IGNORECASE),
    re.compile(r'\bcount\s+(?:the\s+)?buildings?\b', re.IGNORECASE),
    re.compile(r'\bnumber\s+of\s+buildings?\b', re.IGNORECASE),
    re.compile(r'\bhow\s+many\s+\w+\s+(?:have|with|contain)\b', re.IGNORECASE),
    re.compile(r'\blist\s+all\s+buildings?\b', re.IGNORECASE),
    re.compile(r'\bwhich\s+buildings?\s+have\b', re.IGNORECASE),
]


def is_counting_query(query: str) -> bool:
    """
    Detect if a query is asking for a count or list of all buildings.

    Args:
        query: User query string

    Returns:
        True if query is a counting query
    """
    query_lower = query.lower().strip()

    # Check patterns
    for pattern in COUNTING_PATTERNS:
        if pattern.search(query_lower):
            return True

    return False


def extract_document_type_from_query(query: str) -> Optional[str]:
    """
    Extract what document type the user is asking about.

    Args:
        query: User query string

    Returns:
        Document type string or None
    """
    query_lower = query.lower()

    # Map query terms to document types
    doc_type_mappings = {
        'fra': 'fire_risk_assessment',
        'fire risk': 'fire_risk_assessment',
        'fire assessment': 'fire_risk_assessment',
        'bms': 'operational_doc',
        'building management': 'operational_doc',
        'operational': 'operational_doc',
        'o&m': 'operational_doc',
        'planon': 'planon_data',
        'property': 'planon_data',
    }

    for term, doc_type in doc_type_mappings.items():
        if term in query_lower:
            return doc_type

    return None


# ============================================================================
# COUNTING FUNCTIONS
# ============================================================================


def count_buildings_by_document_type(
    doc_type: str,
    index_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Count unique buildings that have documents of a specific type.
    Uses metadata queries to get comprehensive counts.

    Args:
        doc_type: Document type to count (e.g., 'fire_risk_assessment', 'operational_doc')
        index_names: Optional list of indexes to search (defaults to TARGET_INDEXES)

    Returns:
        Dictionary with count results and building list
    """
    if index_names is None:
        index_names = TARGET_INDEXES

    all_buildings: Set[str] = set()
    unique_documents: Set[str] = set()  # Track unique document keys
    # Building -> set of document keys
    documents_by_building: Dict[str, Set[str]] = {}

    for idx_name in index_names:
        try:
            idx = open_index(idx_name)

            # Query with document_type filter to get all matching vectors
            # Use a dummy vector since we only care about metadata
            dummy_vector = [0.0] * get_index_config(idx_name)['dimension']

            # Cast filter to Any to satisfy type checker
            filter_dict: Any = {"document_type": {"$eq": doc_type}}

            results = idx.query(
                vector=dummy_vector,
                filter=filter_dict,
                top_k=10000,  # Large number to get all results
                namespace=DEFAULT_NAMESPACE,
                include_metadata=True
            )

            matches = normalise_matches(results)

            # Extract unique building names and document keys
            for match in matches:
                metadata = match.get('metadata', {})

                # Get document key (unique identifier for the document)
                doc_key = metadata.get('key') or metadata.get('original_file')
                if not doc_key:
                    continue  # Skip if no key found

                # Check multiple building name fields
                building_name = None
                for field in BUILDING_METADATA_FIELDS:
                    value = metadata.get(field)
                    if value:
                        if isinstance(value, list) and value:
                            building_name = str(value[0]).strip()
                        elif isinstance(value, str):
                            building_name = value.strip()

                        if building_name and building_name != 'Unknown':
                            break

                if building_name and building_name != 'Unknown':
                    all_buildings.add(building_name)
                    unique_documents.add(doc_key)

                    # Track documents per building
                    if building_name not in documents_by_building:
                        documents_by_building[building_name] = set()
                    documents_by_building[building_name].add(doc_key)

            logging.info(
                "Index '%s': Found %d unique documents of type '%s' across %d buildings",
                idx_name, len(unique_documents), doc_type, len(all_buildings)
            )

        except Exception as e:  # pylint: disable=broad-except
            logging.warning("Error counting in index '%s': %s", idx_name, e)
            continue

    # Convert sets to counts for output
    doc_counts_by_building = {
        building: len(doc_keys)
        for building, doc_keys in documents_by_building.items()
    }

    # Convert sets to lists for output (to show actual document names)
    doc_keys_by_building = {
        building: sorted(list(doc_keys))
        for building, doc_keys in documents_by_building.items()
    }

    return {
        'document_type': doc_type,
        'unique_buildings': sorted(list(all_buildings)),
        'building_count': len(all_buildings),
        'total_documents': len(unique_documents),  # Unique document count
        'documents_by_building': doc_counts_by_building,
        'document_keys_by_building': doc_keys_by_building,  # NEW: Actual document names
        'indexes_searched': index_names
    }


def count_all_buildings(index_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Count all unique buildings across all document types.

    Args:
        index_names: Optional list of indexes to search (defaults to TARGET_INDEXES)

    Returns:
        Dictionary with count results
    """
    if index_names is None:
        index_names = TARGET_INDEXES

    all_buildings: Set[str] = set()
    buildings_by_doc_type: Dict[str, Set[str]] = {}

    for idx_name in index_names:
        try:
            idx = open_index(idx_name)

            # Query without document type filter to get all documents
            dummy_vector = [0.0] * get_index_config(idx_name)['dimension']

            results = idx.query(
                vector=dummy_vector,
                top_k=10000,  # Large number
                namespace=DEFAULT_NAMESPACE,
                include_metadata=True
            )

            matches = normalise_matches(results)

            for match in matches:
                metadata = match.get('metadata', {})

                # Get building name
                building_name = None
                for field in BUILDING_METADATA_FIELDS:
                    value = metadata.get(field)
                    if value:
                        if isinstance(value, list) and value:
                            building_name = str(value[0]).strip()
                        elif isinstance(value, str):
                            building_name = value.strip()

                        if building_name and building_name != 'Unknown':
                            break

                if building_name and building_name != 'Unknown':
                    all_buildings.add(building_name)

                    # Track by document type
                    doc_type = metadata.get('document_type', 'unknown')
                    if doc_type not in buildings_by_doc_type:
                        buildings_by_doc_type[doc_type] = set()
                    buildings_by_doc_type[doc_type].add(building_name)

        except Exception as e:  # pylint: disable=broad-except
            logging.warning("Error counting in index '%s': %s", idx_name, e)
            continue

    # Convert sets to sorted lists
    buildings_by_doc_type_lists = {
        doc_type: sorted(list(buildings))
        for doc_type, buildings in buildings_by_doc_type.items()
    }

    return {
        'total_buildings': len(all_buildings),
        'all_buildings': sorted(list(all_buildings)),
        'buildings_by_document_type': buildings_by_doc_type_lists,
        'indexes_searched': index_names
    }


# ============================================================================
# ANSWER GENERATION FOR COUNTING QUERIES
# ============================================================================


def generate_counting_answer(query: str) -> Optional[str]:
    """
    Generate an answer for counting queries.

    Args:
        query: User query string

    Returns:
        Answer string or None if not a counting query
    """
    if not is_counting_query(query):
        return None

    # Detect document type
    doc_type = extract_document_type_from_query(query)

    if doc_type:
        # Count buildings with specific document type
        results = count_buildings_by_document_type(doc_type)

        # Map doc_type to friendly name
        doc_type_names = {
            'fire_risk_assessment': 'Fire Risk Assessments (FRAs)',
            'operational_doc': 'BMS/Operational documents',
            'planon_data': 'Planon property records'
        }
        doc_name = doc_type_names.get(doc_type, doc_type)

        building_count = results['building_count']
        total_docs = results['total_documents']

        answer = f"**{building_count} buildings** have {doc_name} in the system.\n\n"
        answer += f"**Total unique documents:** {total_docs}\n\n"

        # List buildings
        if building_count > 0 and building_count <= 50:
            answer += "**Buildings with " + doc_name + ":**\n"
            for building in results['unique_buildings']:
                doc_count = results['documents_by_building'].get(building, 0)
                doc_keys = results['document_keys_by_building'].get(
                    building, [])

                answer += f"- **{building}** ({doc_count} document(s))\n"

                # Show document names for this building
                if doc_keys:
                    for doc_key in doc_keys:
                        answer += f"  - `{doc_key}`\n"

        elif building_count > 50:
            answer += f"**First 50 buildings with {doc_name}:**\n"
            for building in results['unique_buildings'][:50]:
                doc_count = results['documents_by_building'].get(building, 0)
                doc_keys = results['document_keys_by_building'].get(
                    building, [])

                answer += f"- **{building}** ({doc_count} document(s))\n"

                # Show document names for this building
                if doc_keys:
                    for doc_key in doc_keys:
                        answer += f"  - `{doc_key}`\n"

            answer += f"\n... and {building_count - 50} more buildings."

        return answer

    else:
        # Count all buildings across all document types
        results = count_all_buildings()

        total_buildings = results['total_buildings']

        answer = f"**{total_buildings} unique buildings** are indexed in the system.\n\n"

        # Breakdown by document type
        buildings_by_type = results['buildings_by_document_type']

        if buildings_by_type:
            answer += "**Breakdown by document type:**\n"
            for doc_type, buildings in buildings_by_type.items():
                doc_type_names = {
                    'fire_risk_assessment': 'Fire Risk Assessments',
                    'operational_doc': 'BMS/Operational',
                    'planon_data': 'Planon Property Data'
                }
                doc_name = doc_type_names.get(doc_type, doc_type)
                answer += f"- {doc_name}: {len(buildings)} buildings\n"

        # List some buildings
        if total_buildings > 0 and total_buildings <= 30:
            answer += "\n**All buildings:**\n"
            for building in results['all_buildings']:
                answer += f"- {building}\n"
        elif total_buildings > 30:
            answer += "\n**Sample of buildings:**\n"
            for building in results['all_buildings'][:30]:
                answer += f"- {building}\n"
            answer += f"\n... and {total_buildings - 30} more."

        return answer


# ============================================================================
# TESTING
# ============================================================================


def test_counting_queries():
    """Test the counting query detection and answers."""
    test_queries = [
        "How many buildings have FRAs?",
        "Count the buildings with fire risk assessments",
        "How many buildings have BMS documentation?",
        "List all buildings",
        "Which buildings have operational documents?",
        "What is the BMS in Senate House?",  # Not a counting query
    ]

    for query in test_queries:
        is_count = is_counting_query(query)
        doc_type = extract_document_type_from_query(query)
        print(f"\nQuery: {query}")
        print(f"Is counting query: {is_count}")
        print(f"Document type: {doc_type}")

        if is_count:
            answer = generate_counting_answer(query)
            if answer:
                print(f"Answer preview: {answer[:200]}...")


if __name__ == "__main__":
    # Run tests
    test_counting_queries()
