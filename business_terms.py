#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Business terminology mapping and query enhancement for domain-specific terms.
Optimised version with pre-compiled patterns and better type safety.
"""

import os
from typing import Dict, List, Tuple, Optional, Any
import re
from dataclasses import dataclass, field

# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class TermInfo:
    """Information about a business term."""
    term_key: str
    full_name: str
    document_type: str
    search_terms: List[str]
    description: str
    variations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            'term': self.term_key,
            'full_name': self.full_name,
            'document_type': self.document_type,
            'search_terms': self.search_terms,
            'description': self.description,
            'variations': self.variations
        }


# ============================================================================
# TERM DEFINITIONS
# ============================================================================

# Core terminology mappings with structured data
TERM_DEFINITIONS = {
    'fra': TermInfo(
        term_key='fra',
        full_name='Fire Risk Assessment',
        document_type='fire_risk_assessment',
        search_terms=['fire risk assessment',
                      'fire safety', 'FRA', 'fire assessment'],
        description='Fire safety evaluation documents',
        variations=['fras', 'f.r.a.', 'fra', 'fire-risk-assessment']
    ),
    'ahu': TermInfo(
        term_key='ahu',
        full_name='Air Handling Unit',
        document_type='operational_doc',
        search_terms=['air handling unit', 'AHU', 'air handler'],
        description='HVAC air distribution equipment',
        variations=['ahus']
    ),
    'bms': TermInfo(
        term_key='bms',
        full_name='Building Management System',
        document_type='operational_doc',
        search_terms=['building management system', 'BMS', 'controls'],
        description='Building control and automation systems',
        variations=['building management', 'bms system']
    ),
    'hvac': TermInfo(
        term_key='hvac',
        full_name='Heating, Ventilation, and Air Conditioning',
        document_type='operational_doc',
        search_terms=['HVAC', 'heating ventilation', 'climate control'],
        description='Building climate control systems',
        variations=['heating ventilation air conditioning']
    ),
    'planon': TermInfo(
        term_key='planon',
        full_name='Planon Property Management',
        document_type='planon_data',
        search_terms=['planon', 'property management', 'property condition'],
        description='Property management and condition assessment data',
        variations=['property data', 'building data']
    ),
    'iq4': TermInfo(
        term_key='iq4',
        full_name='IQ4 Controller',
        document_type='operational_doc',
        search_terms=['IQ4', 'iq4 controller', 'trend controller'],
        description='Building management system controller',
        variations=['iq-4', 'iq 4']
    ),
    'o&m': TermInfo(
        term_key='o&m',
        full_name='Operations & Maintenance',
        document_type='operational_doc',
        search_terms=['operations maintenance', 'O&M', 'operating manual'],
        description='Operations and maintenance documentation',
        variations=['o and m', 'operations and maintenance', 'om']
    ),
    'desops': TermInfo(
        term_key='desops',
        full_name='Description of Operations',
        document_type='operational_doc',
        search_terms=['description of operations',
                      'DesOps', 'system operations'],
        description='System operation descriptions',
        variations=['des ops', 'des-ops']
    ),
}


# ============================================================================
# PRE-COMPILED PATTERNS
# ============================================================================

def _compile_patterns_for_term(term_info: TermInfo) -> List[re.Pattern]:
    """
    Create pre-compiled regex patterns for a term and its variations.

    Args:
        term_info: Term information object

    Returns:
        List of compiled patterns
    """
    patterns = []

    # Main term pattern
    patterns.append(
        re.compile(rf'\b{re.escape(term_info.term_key)}\b', re.IGNORECASE)
    )

    # Variation patterns
    for variation in term_info.variations:
        patterns.append(
            re.compile(rf'\b{re.escape(variation)}\b', re.IGNORECASE)
        )

    return patterns


# Build pattern cache at module load time
_PATTERN_CACHE: Dict[str, List[re.Pattern]] = {
    term_key: _compile_patterns_for_term(term_info)
    for term_key, term_info in TERM_DEFINITIONS.items()
}


# ============================================================================
# BUSINESS TERM MAPPER
# ============================================================================


class BusinessTermMapper:
    """Maps business terms/acronyms to their technical equivalents for search."""

    # Expose term definitions for external access
    TERM_MAPPINGS = {
        key: info.to_dict()
        for key, info in TERM_DEFINITIONS.items()
    }

    @classmethod
    def detect_business_terms(cls, query: str) -> List[Dict[str, Any]]:
        """
        Detect ALL business terms in a query.

        Args:
            query: User query string

        Returns:
            List of detected term dictionaries
        """
        if not query:
            return []

        query_lower = query.lower()
        detected_terms = []
        seen_terms = set()

        for term_key, patterns in _PATTERN_CACHE.items():
            if term_key in seen_terms:
                continue

            # Check if any pattern matches
            for pattern in patterns:
                if pattern.search(query_lower):
                    term_info = TERM_DEFINITIONS[term_key]
                    detected_terms.append(term_info.to_dict())
                    seen_terms.add(term_key)
                    break  # Found this term, move to next

        return detected_terms

    @classmethod
    def enhance_query_with_terms(cls, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Enhance query with business term expansions.

        Args:
            query: Original user query

        Returns:
            (enhanced_query, term_context)
            - enhanced_query: Query with expanded terms
            - term_context: Dictionary mapping term keys to term info
        """
        if not query:
            return query, {}

        detected = cls.detect_business_terms(query)

        if not detected:
            return query, {}

        # Build enhanced query
        enhanced_parts = [query]
        term_context = {}

        for term_dict in detected:
            # Add search terms that aren't already in the query
            for search_term in term_dict['search_terms']:
                if search_term.lower() not in query.lower():
                    enhanced_parts.append(search_term)

            term_context[term_dict['term']] = term_dict

        enhanced_query = ' '.join(enhanced_parts)
        return enhanced_query, term_context

    @classmethod
    def get_term_info(cls, term_key: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific term.

        Args:
            term_key: Term key to look up

        Returns:
            Term info dictionary or None if not found
        """
        term_info = TERM_DEFINITIONS.get(term_key.lower())
        return term_info.to_dict() if term_info else None

    @classmethod
    def get_all_terms(cls) -> List[str]:
        """
        Get list of all recognised term keys.

        Returns:
            List of term keys
        """
        return list(TERM_DEFINITIONS.keys())

    @classmethod
    def get_terms_by_document_type(cls, doc_type: str) -> List[Dict[str, Any]]:
        """
        Get all terms associated with a specific document type.

        Args:
            doc_type: Document type to filter by

        Returns:
            List of term dictionaries
        """
        return [
            term_info.to_dict()
            for term_info in TERM_DEFINITIONS.values()
            if term_info.document_type == doc_type
        ]

    @classmethod
    def expand_acronym(cls, acronym: str) -> Optional[str]:
        """
        Expand an acronym to its full name.

        Args:
            acronym: Acronym to expand

        Returns:
            Full name or None if not found
        """
        term_info = TERM_DEFINITIONS.get(acronym.lower())
        return term_info.full_name if term_info else None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def add_custom_term(
    term_key: str,
    full_name: str,
    document_type: str,
    search_terms: List[str],
    description: str,
    variations: Optional[List[str]] = None
) -> bool:
    """
    Add a custom business term at runtime.

    Args:
        term_key: Unique key for the term
        full_name: Full name/expansion
        document_type: Associated document type
        search_terms: List of search terms
        description: Human-readable description
        variations: Optional list of variations

    Returns:
        True if added successfully, False if term already exists
    """
    term_key_lower = term_key.lower()

    if term_key_lower in TERM_DEFINITIONS:
        return False

    # Create term info
    term_info = TermInfo(
        term_key=term_key_lower,
        full_name=full_name,
        document_type=document_type,
        search_terms=search_terms,
        description=description,
        variations=variations or []
    )

    # Add to definitions
    TERM_DEFINITIONS[term_key_lower] = term_info

    # Update pattern cache
    _PATTERN_CACHE[term_key_lower] = _compile_patterns_for_term(term_info)

    # Update mapper's term mappings
    BusinessTermMapper.TERM_MAPPINGS[term_key_lower] = term_info.to_dict()

    return True


def get_term_statistics() -> Dict[str, Any]:
    """
    Get statistics about registered terms.

    Returns:
        Dictionary with statistics
    """
    doc_type_counts = {}
    total_variations = 0
    total_search_terms = 0

    for term_info in TERM_DEFINITIONS.values():
        # Count by document type
        doc_type_counts[term_info.document_type] = \
            doc_type_counts.get(term_info.document_type, 0) + 1

        # Count variations and search terms
        total_variations += len(term_info.variations)
        total_search_terms += len(term_info.search_terms)

    return {
        'total_terms': len(TERM_DEFINITIONS),
        'by_document_type': doc_type_counts,
        'total_variations': total_variations,
        'total_search_terms': total_search_terms,
        'avg_variations_per_term': total_variations / len(TERM_DEFINITIONS) if TERM_DEFINITIONS else 0,
        'avg_search_terms_per_term': total_search_terms / len(TERM_DEFINITIONS) if TERM_DEFINITIONS else 0
    }


def validate_term_definitions() -> List[str]:
    """
    Validate all term definitions for consistency.

    Returns:
        List of validation warnings (empty if all valid)
    """
    warnings = []

    for term_key, term_info in TERM_DEFINITIONS.items():
        # Check for empty fields
        if not term_info.full_name:
            warnings.append(f"Term '{term_key}' has empty full_name")

        if not term_info.search_terms:
            warnings.append(f"Term '{term_key}' has no search_terms")

        if not term_info.description:
            warnings.append(f"Term '{term_key}' has empty description")

        # Check for duplicate variations
        if len(term_info.variations) != len(set(term_info.variations)):
            warnings.append(f"Term '{term_key}' has duplicate variations")

    return warnings


# Run validation at module load (in development)
if os.getenv('VALIDATE_BUSINESS_TERMS', '').lower() == 'true':
    validation_warnings = validate_term_definitions()
    if validation_warnings:
        import logging
        for warning in validation_warnings:
            logging.warning("Business term validation: %s", warning)
