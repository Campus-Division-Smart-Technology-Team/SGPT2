#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query classification to determine if queries need search or direct response.
Optimised version with pre-compiled patterns and type hints.
"""

import re
import random
from typing import Optional, Tuple, Dict, Any, Literal, List
from dataclasses import dataclass, field

from building_utils import extract_building_from_query
from business_terms import BusinessTermMapper

# ============================================================================
# TYPES
# ============================================================================

QueryType = Literal['greeting', 'about', 'gratitude', 'farewell', 'search']

# ============================================================================
# CONSTANTS
# ============================================================================

# Emojis (properly encoded)
EMOJI_GORILLA = "🦍"
EMOJI_BUILDING = "🏢"
EMOJI_FIRE = "🔥"

# ============================================================================
# PRE-COMPILED PATTERNS
# ============================================================================

# Greeting patterns (pre-compiled for performance)
GREETING_PATTERNS = [
    re.compile(
        r'^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening)|howdy)[\s!.,]*$', re.IGNORECASE),
    re.compile(
        r'^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening)|howdy)\s+alfred[\s!.,]*$', re.IGNORECASE),
    re.compile(r'^alfred[\s!.,]*$', re.IGNORECASE)
]

# About Alfred patterns
ABOUT_ALFRED_PATTERNS = [
    re.compile(r'who\s+are\s+you', re.IGNORECASE),
    re.compile(r'what\s+are\s+you', re.IGNORECASE),
    re.compile(r'tell\s+me\s+about\s+yourself', re.IGNORECASE),
    re.compile(r'what\s+can\s+you\s+(do|help)', re.IGNORECASE),
    re.compile(r'how\s+can\s+you\s+(help|assist)', re.IGNORECASE),
    re.compile(r'what\s+do\s+you\s+know\s+about', re.IGNORECASE),
    re.compile(r'your\s+(capabilities|functions|abilities)', re.IGNORECASE),
    re.compile(r'what\s+is\s+your\s+purpose', re.IGNORECASE)
]

# Gratitude patterns
GRATITUDE_PATTERNS = [
    re.compile(
        r'^(thank\s*you|thanks|cheers|ta|much\s+appreciated)[\s!.,]*$', re.IGNORECASE),
    re.compile(
        r'^(great|awesome|perfect|excellent|brilliant)[\s!.,]*$', re.IGNORECASE),
    re.compile(
        r'^(that\'s\s+helpful|very\s+helpful|that\s+helps)[\s!.,]*$', re.IGNORECASE)
]

# Farewell patterns
FAREWELL_PATTERNS = [
    re.compile(
        r'^(bye|goodbye|see\s+you|farewell|take\s+care|have\s+a\s+good\s+day)[\s!.,]*$', re.IGNORECASE),
    re.compile(
        r'^(thanks\s+and\s+bye|bye\s+for\s+now)[\s!.,]*$', re.IGNORECASE),
    re.compile(
        r'^(bye|goodbye|see\s+you|farewell|take\s+care|have\s+a\s+good\s+day)\s+alfred[\s!.,]*$', re.IGNORECASE),
    re.compile(
        r'^alfred\s+(bye|goodbye|see\s+you|farewell|take\s+care|have\s+a\s+good\s+day)[\s!.,]*$', re.IGNORECASE)
]

# Action detection patterns
ACTION_PATTERNS = {
    'summarise': re.compile(r'\b(summarise|summarize|summary|overview)\b', re.IGNORECASE),
    'list': re.compile(r'\b(list|show|display|what are)\b', re.IGNORECASE),
    'find': re.compile(r'\b(find|search|locate|where|get)\b', re.IGNORECASE),
    'compare': re.compile(r'\b(compare|difference|versus|vs)\b', re.IGNORECASE)
}

# ============================================================================
# RESPONSE TEMPLATES
# ============================================================================

GREETING_RESPONSES = [
    f"Hello! I'm Alfred {EMOJI_GORILLA}, your helpful assistant at the University of Bristol. I can help you find information about:\n\n• {EMOJI_BUILDING} Building Management Systems (BMS)\n• {EMOJI_FIRE} Fire Risk Assessments (FRAs)\n\nWhat would you like to know today?",
    "Hi there! I'm Alfred, ready to help you search through our knowledge bases. Feel free to ask me about BMS and FRAs. How can I assist you?",
    f"Hello! Alfred here {EMOJI_GORILLA}, your University of Bristol assistant. I have access to information about building management systems and Fire Risk Assessments. What can I help you with?"
]

ABOUT_RESPONSE = f"""I'm Alfred {EMOJI_GORILLA}, a specialised assistant for the University of Bristol's Smart Technology team.

**What I can do:**
- Search and retrieve information from our knowledge bases
- Provide information about Building Management Systems (BMS) and Fire Risk Assessments (FRAs) at the university
- Tell you when documents were last updated or published

**How to use me:**
Simply type your question in natural language. I'll search through the relevant indexes and provide you with:
- A comprehensive answer based on the available information
- The publication or update date of the source material
- Links to view the raw search results if you need more detail

**Note:** My knowledge is limited to what's in the indexed documents. If I can't find something, I'll let you know honestly rather than making things up.
"""

GRATITUDE_RESPONSES = [
    "You're welcome! Is there anything else I can help you find?",
    "Happy to help! Feel free to ask if you need more information.",
    "Glad I could assist! Let me know if you have any other questions.",
    "My pleasure! I'm here if you need to search for anything else."
]

FAREWELL_RESPONSES = [
    "Goodbye! Feel free to come back anytime you need information about BMS or FRAs across the different UoB buildings.",
    "Take care! I'll be here whenever you need to search our knowledge bases.",
    "See you later! Don't hesitate to return if you have more questions.",
    f"Farewell! Have a great day at the University of Bristol! {EMOJI_GORILLA}"
]


# ============================================================================
# QUERY INTENT DATA CLASS
# ============================================================================


@dataclass
class QueryIntent:
    """Structured representation of query intent."""
    action: Optional[str] = None
    building: Optional[str] = None
    document_type: Optional[str] = None
    business_terms: List[Dict[str, Any]] = field(default_factory=list)
    original_query: str = ""


# ============================================================================
# QUERY CLASSIFIER
# ============================================================================


class QueryClassifier:
    """Classify user queries to determine if they need index search or direct response."""

    @classmethod
    def classify_query(cls, query: str) -> Tuple[QueryType, Optional[str]]:
        """
        Classify a query and return (query_type, suggested_response).

        Args:
            query: User query string

        Returns:
            query_type: One of 'greeting', 'about', 'gratitude', 'farewell', 'search'
            suggested_response: Pre-defined response for non-search queries, None for search
        """
        query_lower = query.lower().strip()

        # Check greeting patterns
        if cls._matches_patterns(query_lower, GREETING_PATTERNS):
            return 'greeting', cls._get_random_response(GREETING_RESPONSES)

        # Check about Alfred patterns
        if cls._matches_patterns(query_lower, ABOUT_ALFRED_PATTERNS):
            return 'about', ABOUT_RESPONSE

        # Check gratitude patterns
        if cls._matches_patterns(query_lower, GRATITUDE_PATTERNS):
            return 'gratitude', cls._get_random_response(GRATITUDE_RESPONSES)

        # Check farewell patterns
        if cls._matches_patterns(query_lower, FAREWELL_PATTERNS):
            return 'farewell', cls._get_random_response(FAREWELL_RESPONSES)

        # Default to search
        return 'search', None

    @staticmethod
    def _matches_patterns(text: str, patterns: list) -> bool:
        """
        Check if text matches any of the given pre-compiled patterns.

        Args:
            text: Text to check
            patterns: List of compiled regex patterns

        Returns:
            True if any pattern matches
        """
        for pattern in patterns:
            if pattern.search(text):
                return True
        return False

    @staticmethod
    def _get_random_response(responses: list, seed: Optional[int] = None) -> str:
        """
        Get a random response from a list.

        Args:
            responses: List of response strings
            seed: Optional random seed for deterministic behavior (useful for testing)

        Returns:
            Random response string
        """
        if seed is not None:
            random.seed(seed)
        return random.choice(responses)


# ============================================================================
# PUBLIC API
# ============================================================================


def should_search_index(query: str) -> Tuple[bool, Optional[str]]:
    """
    Determine if a query requires an index search or can be answered directly.

    Args:
        query: User query string

    Returns:
        (should_search, direct_response)
        - should_search: True if query needs search, False for direct response
        - direct_response: Response string for non-search queries, None for search
    """
    query_type, suggested_response = QueryClassifier.classify_query(query)

    if query_type == 'search':
        return True, None
    else:
        return False, suggested_response


def parse_user_intent(query: str) -> QueryIntent:
    """
    Parse user query to extract intent, building, document type, and action.

    Args:
        query: User query string

    Returns:
        QueryIntent object with extracted information
    """
    intent = QueryIntent(original_query=query)

    # Detect action verbs
    for action, pattern in ACTION_PATTERNS.items():
        if pattern.search(query):
            intent.action = action
            break

    # Extract building
    intent.building = extract_building_from_query(query)

    # Detect business terms
    term_mappings = BusinessTermMapper.detect_business_terms(query)
    if term_mappings:
        intent.business_terms = term_mappings
        # Set document type from first detected term
        intent.document_type = term_mappings[0].get('document_type')

    return intent


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_query_complexity(query: str) -> str:
    """
    Estimate query complexity for routing/optimisation decisions.

    Args:
        query: User query string

    Returns:
        Complexity level: 'simple', 'medium', 'complex'
    """
    # Simple heuristics
    word_count = len(query.split())
    has_building = extract_building_from_query(query) is not None
    has_terms = bool(BusinessTermMapper.detect_business_terms(query))

    if word_count <= 5 and not has_building:
        return 'simple'
    elif word_count <= 15 or (has_building and not has_terms):
        return 'medium'
    else:
        return 'complex'


def is_question(query: str) -> bool:
    """
    Determine if query is a question (useful for response formatting).

    Args:
        query: User query string

    Returns:
        True if query appears to be a question
    """
    query_lower = query.lower().strip()

    # Check for question marks
    if '?' in query:
        return True

    # Check for question words at start
    question_words = ['what', 'when', 'where', 'who', 'why', 'how',
                      'which', 'can', 'could', 'would', 'should', 'is', 'are', 'does', 'do']
    first_word = query_lower.split()[0] if query_lower.split() else ''

    return first_word in question_words


def extract_keywords(query: str) -> list:
    """
    Extract key terms from query for enhanced search.

    Args:
        query: User query string

    Returns:
        List of keywords
    """
    # Simple keyword extraction (could be enhanced with NLP)
    # Remove common stop words
    stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'what', 'when', 'where', 'who', 'why'
    }

    words = query.lower().split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2]

    return keywords


# ============================================================================
# TESTING/DEBUG UTILITIES
# ============================================================================


def test_classifier(test_queries: list) -> Dict[str, list]:
    """
    Test the classifier with a list of queries (useful for validation).

    Args:
        test_queries: List of query strings to test

    Returns:
        Dictionary mapping query types to lists of queries
    """
    results = {
        'greeting': [],
        'about': [],
        'gratitude': [],
        'farewell': [],
        'search': []
    }

    for query in test_queries:
        query_type, _ = QueryClassifier.classify_query(query)
        results[query_type].append(query)

    return results


def get_classifier_stats() -> Dict[str, int]:
    """
    Get statistics about the classifier configuration.

    Returns:
        Dictionary with pattern counts
    """
    return {
        'greeting_patterns': len(GREETING_PATTERNS),
        'about_patterns': len(ABOUT_ALFRED_PATTERNS),
        'gratitude_patterns': len(GRATITUDE_PATTERNS),
        'farewell_patterns': len(FAREWELL_PATTERNS),
        'action_patterns': len(ACTION_PATTERNS),
        'total_patterns': (
            len(GREETING_PATTERNS) +
            len(ABOUT_ALFRED_PATTERNS) +
            len(GRATITUDE_PATTERNS) +
            len(FAREWELL_PATTERNS) +
            len(ACTION_PATTERNS)
        )
    }
