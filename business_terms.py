#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Business terminology mapping and query enhancement for domain-specific terms.
"""

from typing import Dict, List, Tuple, Optional
import re


class BusinessTermMapper:
    """Maps business terms/acronyms to their technical equivalents for search."""

    # Core terminology mappings
    TERM_MAPPINGS = {
        'fra': {
            'full_name': 'Fire Risk Assessment',
            'document_type': 'fire_risk_assessment',
            'search_terms': ['fire risk assessment', 'fire safety', 'FRA', 'fire assessment'],
            'variations': ['fras', 'f.r.a.', 'fire-risk-assessment'],
            'description': 'Fire safety evaluation documents'
        },
        'AHU': {
            'full_name': 'Air Handling Unit',
            'document_type': 'operational_doc',
            'search_terms': ['air handling unit', 'AHU', 'air handler'],
            'description': 'HVAC air distribution equipment'
        },
        'BMS': {
            'full_name': 'Building Management System',
            'document_type': 'operational_doc',
            'search_terms': ['building management system', 'BMS', 'controls'],
            'description': 'Building control and automation systems'
        },
        'HVAC': {
            'full_name': 'Heating, Ventilation, and Air Conditioning',
            'document_type': 'operational_doc',
            'search_terms': ['HVAC', 'heating ventilation', 'climate control'],
            'description': 'Building climate control systems'
        },
        'Planon': {
            'full_name': 'Planon Property Management',
            'document_type': 'planon_data',
            'search_terms': ['planon', 'property management', 'property condition'],
            'variations': ['property data', 'building data'],
            'description': 'Property management and condition assessment data'
        },
        'iq4': {
            'full_name': 'IQ4 Controller',
            'document_type': 'operational_doc',
            'search_terms': ['IQ4', 'iq4 controller', 'trend controller'],
            'description': 'Building management system controller'
        },
        'O&M': {
            'full_name': 'Operations & Maintenance',
            'document_type': 'operational_doc',
            'search_terms': ['operations maintenance', 'O&M', 'operating manual'],
            'description': 'Operations and maintenance documentation'
        },
        'Desops': {
            'full_name': 'Description of Operations',
            'document_type': 'operational_doc',
            'search_terms': ['description of operations', 'DesOps', 'system operations'],
            'description': 'System operation descriptions'
        }
    }

    @classmethod
    def detect_business_terms(cls, query: str) -> List[Dict]:
        """Detect ALL business terms in a query (not just the first one)."""
        query_lower = query.lower()
        detected_terms = []
        seen_terms = set()  # Avoid duplicates

        for term_key, term_info in cls.TERM_MAPPINGS.items():
            if term_key in seen_terms:
                continue

            # Check main term and variations
            patterns = [rf'\b{re.escape(term_key)}\b']
            if 'variations' in term_info:
                patterns.extend([rf'\b{re.escape(var)}\b'
                                 for var in term_info['variations']])

            for pattern in patterns:
                if re.search(pattern, query_lower):
                    detected_terms.append({
                        'term': term_key,
                        **term_info
                    })
                    seen_terms.add(term_key)
                    break

        return detected_terms

    @classmethod
    def enhance_query_with_terms(cls, query: str) -> Tuple[str, Dict]:
        """
        Enhance query with business term expansions.
        Returns (enhanced_query, term_context)
        """
        detected = cls.detect_business_terms(query)

        if not detected:
            return query, {}

        # Build enhanced query
        enhanced_parts = [query]
        term_context = {}

        for term_info in detected:
            # Add search terms to query
            for search_term in term_info['search_terms']:
                if search_term.lower() not in query.lower():
                    enhanced_parts.append(search_term)

            term_context[term_info['term']] = term_info

        enhanced_query = ' '.join(enhanced_parts)
        return enhanced_query, term_context
