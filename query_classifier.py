#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query classification to determine if queries need search or direct response.
"""

import re
import random
from typing import Optional, Tuple, Dict, Any

from building_utils import extract_building_from_query
from business_terms import BusinessTermMapper


class QueryClassifier:
    """Classify user queries to determine if they need index search or direct response."""

    # Patterns that don't require search
    GREETING_PATTERNS = [
        r'^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening)|howdy)[\s!.,]*$',
        r'^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening)|howdy)\s+alfred[\s!.,]*$',
        r'^alfred[\s!.,]*$'
    ]

    ABOUT_ALFRED_PATTERNS = [
        r'who\s+are\s+you',
        r'what\s+are\s+you',
        r'tell\s+me\s+about\s+yourself',
        r'what\s+can\s+you\s+(do|help)',
        r'how\s+can\s+you\s+(help|assist)',
        r'what\s+do\s+you\s+know\s+about',
        r'your\s+(capabilities|functions|abilities)',
        r'what\s+is\s+your\s+purpose'
    ]

    GRATITUDE_PATTERNS = [
        r'^(thank\s*you|thanks|cheers|ta|much\s+appreciated)[\s!.,]*$',
        r'^(great|awesome|perfect|excellent|brilliant)[\s!.,]*$',
        r'^(that\'s\s+helpful|very\s+helpful|that\s+helps)[\s!.,]*$'
    ]

    FAREWELL_PATTERNS = [
        r'^(bye|goodbye|see\s+you|farewell|take\s+care|have\s+a\s+good\s+day)[\s!.,]*$',
        r'^(thanks\s+and\s+bye|bye\s+for\s+now)[\s!.,]*$',
        r'^(bye|goodbye|see\s+you|farewell|take\s+care|have\s+a\s+good\s+day)\s+alfred[\s!.,]*$',
        r'^alfred\s+(bye|goodbye|see\s+you|farewell|take\s+care|have\s+a\s+good\s+day)[\s!.,]*$'
    ]

    @classmethod
    def classify_query(cls, query: str) -> Tuple[str, Optional[str]]:
        """
        Classify a query and return (query_type, suggested_response).

        Returns:
            query_type: One of 'greeting', 'about', 'gratitude', 'farewell', 'search'
            suggested_response: Pre-defined response for non-search queries, None for search
        """
        query_lower = query.lower().strip()

        # Check greeting patterns
        for pattern in cls.GREETING_PATTERNS:
            if re.match(pattern, query_lower):
                return 'greeting', cls.get_greeting_response()

        # Check about Alfred patterns
        for pattern in cls.ABOUT_ALFRED_PATTERNS:
            if re.search(pattern, query_lower):
                return 'about', cls.get_about_response()

        # Check gratitude patterns
        for pattern in cls.GRATITUDE_PATTERNS:
            if re.match(pattern, query_lower):
                return 'gratitude', cls.get_gratitude_response()

        # Check farewell patterns
        for pattern in cls.FAREWELL_PATTERNS:
            if re.match(pattern, query_lower):
                return 'farewell', cls.get_farewell_response()

        # Default to search
        return 'search', None

    @staticmethod
    def get_greeting_response() -> str:
        """Return a greeting response."""
        greetings = [
            "Hello! I'm Alfred 🦍, your helpful assistant at the University of Bristol. I can help you find information about:\n\n• 🍎 Apples (the fruit) and Apple Inc.\n\n• 🏢 Building Management Systems (BMS)\n\n• 🔥 Fire Risk Assessments (FRAs)\n\nWhat would you like to know today?",
            "Hi there! I'm Alfred, ready to help you search through our knowledge bases. Feel free to ask me about apples, BMS and FRAs. How can I assist you?",
            "Hello! Alfred here, your University of Bristol assistant. I have access to information about apples, building management systems and Fire Risk Assessments. What can I help you with?"
        ]
        return random.choice(greetings)

    @staticmethod
    def get_about_response() -> str:
        """Return information about Alfred."""
        return """
        I'm Alfred 🦍, a specialised assistant for the University of Bristol's Smart Technology team.

        **What I can do:**
        - Search and retrieve information from our knowledge bases
        - Answer questions about apples (both the fruit and Apple Inc.)
        - Provide information about Building Management Systems (BMS) and Fire Risk Assessments (FRAs) at the university
        - Tell you when documents were last updated or published
        
        **How to use me:**
        Simply type your question in natural language. I'll search through the relevant indexes and provide you with:
        - A comprehensive answer based on the available information
        - The publication or update date of the source material
        - Links to view the raw search results if you need more detail
        
        **Note:** My knowledge is limited to what's in the indexed documents. If I can't find something, I'll let you know honestly rather than making things up."""

    @staticmethod
    def get_gratitude_response() -> str:
        """Return a response to gratitude."""
        responses = [
            "You're welcome! Is there anything else I can help you find?",
            "Happy to help! Feel free to ask if you need more information.",
            "Glad I could assist! Let me know if you have any other questions.",
            "My pleasure! I'm here if you need to search for anything else."
        ]
        return random.choice(responses)

    @staticmethod
    def get_farewell_response() -> str:
        """Return a farewell response."""
        responses = [
            "Goodbye! Feel free to come back anytime you need information about apples or BMS systems.",
            "Take care! I'll be here whenever you need to search our knowledge bases.",
            "See you later! Don't hesitate to return if you have more questions.",
            "Farewell! Have a great day at the University of Bristol! 🦍"
        ]
        return random.choice(responses)


def should_search_index(query: str) -> Tuple[bool, Optional[str]]:
    """
    Determine if a query requires an index search or can be answered directly.

    Returns:
        (should_search, direct_response)
    """
    query_type, suggested_response = QueryClassifier.classify_query(query)

    if query_type == 'search':
        return True, None
    else:
        return False, suggested_response


def parse_user_intent(query: str) -> Dict[str, Any]:
    """
    Parse user query to extract intent, building, document type, and action.
    """
    intent = {
        'action': None,  # 'summarise', 'list', 'find', 'compare'
        'building': None,
        'document_type': None,
        'business_terms': [],
        'original_query': query
    }

    # Detect action verbs
    action_patterns = {
        'summarise': r'\b(summarise|summarize|summary|overview)\b',
        'list': r'\b(list|show|display|what are)\b',
        'find': r'\b(find|search|locate|where|get)\b',
        'compare': r'\b(compare|difference|versus|vs)\b'
    }

    for action, pattern in action_patterns.items():
        if re.search(pattern, query.lower()):
            intent['action'] = action
            break

    # Extract building
    intent['building'] = extract_building_from_query(query)

    # Detect business terms
    term_mappings = BusinessTermMapper.detect_business_terms(query)
    if term_mappings:
        intent['business_terms'] = term_mappings
        # Set document type from first detected term
        intent['document_type'] = term_mappings[0]['document_type']

    return intent
