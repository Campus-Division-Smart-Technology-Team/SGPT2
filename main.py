#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Streamlit application for Alfred the Gorilla chatbot.
IMPROVED VERSION: Dynamic building cache initialisation across all indexes.
"""

import logging
from typing import Dict, List, Any, Optional
import streamlit as st

# Import our modules
from ui_components import (
    setup_page_config, render_custom_css, render_header, render_tabs,
    render_sidebar, display_publication_date_info,
    display_low_score_warning, initialise_chat_history, display_chat_history
)
from query_classifier import should_search_index
from search_operations import perform_federated_search
from building_utils import (
    populate_building_cache_from_multiple_indexes, get_cache_status,
    get_indexes_with_buildings
)
from pinecone_utils import open_index
from config import TARGET_INDEXES, DEFAULT_NAMESPACE

# ============================================================================
# CONSTANTS
# ============================================================================

# Emojis (properly encoded)
EMOJI_GORILLA = "🦍"
EMOJI_MEDAL = "🥇"
EMOJI_BOOKS = "📚"

# UI text
NO_RESULTS_MESSAGE = "I couldn't find any relevant information in our knowledge bases. Regan has told me to say I don't know."
ERROR_MESSAGE_TEMPLATE = "Sorry, I encountered an error while searching: {error}"
SEARCH_SPINNER_TEXT = "Searching across indexes and analysing document dates..."

# Input validation
MAX_QUERY_LENGTH = 1000
MIN_QUERY_LENGTH = 2

# Setup logging
logging.basicConfig(level=logging.INFO)


# ============================================================================
# INITIALISATION
# ============================================================================


@st.cache_resource
def initialise_building_cache():
    """
    Initialise building name cache from ALL Pinecone indexes.
    IMPROVED: Tries all indexes and aggregates results.

    Returns:
        Dictionary with cache status
    """
    try:
        # Check if already populated
        cache_status = get_cache_status()
        if cache_status['populated']:
            logging.info(
                "Building cache already populated, skipping initialisation")
            return cache_status

        # Try to populate from ALL indexes (not just the first one)
        logging.info(
            "Initialising building cache from %d indexes...", len(TARGET_INDEXES))

        results = populate_building_cache_from_multiple_indexes(
            TARGET_INDEXES,
            DEFAULT_NAMESPACE
        )

        # Check final cache status
        cache_status = get_cache_status()

        if cache_status['populated']:
            indexes_with_data = cache_status.get('indexes_with_buildings', [])
            logging.info(
                "✅ Building cache initialised: %d canonical names, %d aliases from %d index(es)",
                cache_status['canonical_names'],
                cache_status['aliases'],
                len(indexes_with_data)
            )

            # Log which indexes have building data
            for idx_name, count in results.items():
                if count > 0:
                    logging.info("  - '%s': %d buildings", idx_name, count)

            return cache_status
        else:
            logging.warning(
                "⚠️  Could not initialise building cache from any of %d indexes",
                len(TARGET_INDEXES)
            )
            return cache_status

    except Exception as e:  # pylint: disable=broad-except
        logging.error("Error initialising building cache: %s",
                      e, exc_info=True)
        return {
            'populated': False,
            'canonical_names': 0,
            'aliases': 0,
            'indexes_with_buildings': []
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def validate_query(query: str) -> tuple[bool, Optional[str]]:
    """
    Validate user query.
    Returns (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, "Please enter a question."

    if len(query.strip()) < MIN_QUERY_LENGTH:
        return False, f"Query too short (minimum {MIN_QUERY_LENGTH} characters)."

    if len(query) > MAX_QUERY_LENGTH:
        return False, f"Query too long (maximum {MAX_QUERY_LENGTH} characters)."

    return True, None


def render_result_item(
    result: Dict[str, Any],
    index: int,
    is_top: bool = False,
    max_snippet_length: int = 500
):
    """
    Render a single search result item.
    """
    if is_top:
        st.markdown(
            f'<div class="top-result-highlight">{EMOJI_MEDAL} <strong>TOP RESULT</strong></div>',
            unsafe_allow_html=True
        )

    # Format score and metadata
    score = result.get('score', 0)
    key = result.get('key', 'Unknown')
    index_name = result.get('index', '?')
    namespace = result.get('namespace', '__default__')

    st.markdown(
        f"**{index}. Score:** {score:.3f}  \n"
        f"_Document:_ `{key}`  •  _Index:_ `{index_name}`  •  _Namespace:_ `{namespace}`"
    )

    # Display text snippet
    snippet = result.get("text") or "_(no text in metadata)_"
    if len(snippet) > max_snippet_length:
        snippet = snippet[:max_snippet_length] + "..."
    st.write(snippet)

    # Display ID as caption
    result_id = result.get('id') or '—'
    st.caption(f"ID: {result_id}")


# ============================================================================
# MAIN APPLICATION
# ============================================================================


def main():
    """Main application function."""
    # Setup page
    setup_page_config()
    render_custom_css()

    # Initialise building cache
    cache_status = initialise_building_cache()

    if not cache_status['populated']:
        st.warning(
            "⚠️ Building name cache could not be initialised. "
            "Building name detection may be limited to pattern matching."
        )
    else:
        indexes_with_buildings = cache_status.get('indexes_with_buildings', [])
        if indexes_with_buildings:
            st.success(
                f"✅ Building data loaded from {len(indexes_with_buildings)} index(es): "
                f"{', '.join(indexes_with_buildings)}"
            )

    render_header()

    # Render main content
    render_tabs()

    # Render sidebar and get settings
    top_k = render_sidebar()

    # Initialise and display chat
    initialise_chat_history()
    display_chat_history()

    # Handle new chat input
    handle_chat_input(top_k)

    # Display last results if they exist
    display_last_results()


def handle_chat_input(top_k: int):
    """Handle new chat input from user."""
    query = st.chat_input("Ask me about BMS or FRAs...")

    if not query:
        return

    # Validate query
    is_valid, error_message = validate_query(query)
    if not is_valid:
        with st.chat_message("assistant", avatar=EMOJI_GORILLA):
            st.warning(error_message)
        return

    # Trim whitespace
    query = query.strip()

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    # Check if we should search or respond directly
    should_search, direct_response = should_search_index(query)

    if not should_search and direct_response:
        # Handle non-search queries (greetings, about, etc.)
        handle_direct_response(direct_response)
    else:
        # Perform search and generate response
        handle_search_query(query, top_k)


def handle_direct_response(response: str):
    """Handle direct responses without search."""
    with st.chat_message("assistant", avatar=EMOJI_GORILLA):
        st.markdown(response)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })


def handle_search_query(query: str, top_k: int):
    """Handle search queries with federated search."""
    with st.chat_message("assistant", avatar=EMOJI_GORILLA):
        with st.spinner(SEARCH_SPINNER_TEXT):
            try:
                results, answer, publication_date_info, score_too_low = perform_federated_search(
                    query, top_k
                )

                # Store results in session state
                st.session_state.last_results = results

                if not results:
                    handle_no_results()
                elif score_too_low:
                    handle_low_score_results(answer, results)
                else:
                    handle_successful_results(
                        answer, results, publication_date_info)

            except Exception as e:  # pylint: disable=broad-except
                handle_search_error(e)


def handle_no_results():
    """Handle case when no results are found."""
    st.markdown(NO_RESULTS_MESSAGE)
    st.session_state.messages.append({
        "role": "assistant",
        "content": NO_RESULTS_MESSAGE
    })


def handle_low_score_results(answer: str, results: List[Dict[str, Any]]):
    """Handle case when results have scores below threshold."""
    st.markdown(answer)
    display_low_score_warning()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "results": results,
        "score_too_low": True
    })


def handle_successful_results(
    answer: str,
    results: List[Dict[str, Any]],
    publication_date_info: str
):
    """Handle successful search results."""
    if answer:
        # Display LLM-generated answer
        st.markdown(answer)

        # Display publication date info prominently
        if publication_date_info:
            display_publication_date_info(publication_date_info)

        # Store message with results and publication date info
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "results": results,
            "publication_date_info": publication_date_info
        })
    else:
        # No answer generation, show results directly
        display_direct_results(results, publication_date_info)


def handle_search_error(error: Exception):  # pylint: disable=broad-except
    """Handle errors during search."""
    error_msg = ERROR_MESSAGE_TEMPLATE.format(error=str(error))
    st.error(error_msg)
    logging.error("Search error: %s", error, exc_info=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": error_msg
    })


def display_direct_results(results: List[Dict[str, Any]], publication_date_info: str):
    """Display search results directly when no LLM answer is generated."""
    response = f"I found {len(results)} relevant results:"
    st.markdown(response)

    # Render each result
    for i, result in enumerate(results, 1):
        render_result_item(result, i, is_top=(i == 1))

        # Add separator between results
        if i < len(results):
            st.markdown("---")

    # Display publication date info
    if publication_date_info:
        display_publication_date_info(publication_date_info)

    # Store in session
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "results": results,
        "publication_date_info": publication_date_info
    })


def display_last_results():
    """Display last search results in expandable section."""
    if "last_results" not in st.session_state or not st.session_state.last_results:
        return

    results = st.session_state.last_results
    result_count = len(results)

    with st.expander(f"{EMOJI_BOOKS} Last Search: {result_count} results", expanded=False):
        for i, result in enumerate(results, 1):
            render_result_item(result, i, is_top=(
                i == 1), max_snippet_length=300)

            # Add separator between results
            if i < result_count:
                st.markdown("---")


# ============================================================================
# ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    main()
