#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Streamlit application for Alfred the Gorilla chatbot.
Enhanced with intelligent query classification and federated search.
"""

import logging
import streamlit as st

# Import our modules
from ui_components import (
    setup_page_config, render_custom_css, render_header, render_tabs,
    render_sidebar, display_publication_date_info,
    display_low_score_warning, initialise_chat_history, display_chat_history
)
from query_classifier import should_search_index
from search_operations import perform_federated_search

# Setup logging
logging.basicConfig(level=logging.INFO)


def main():
    """Main application function."""
    # Setup page
    setup_page_config()
    render_custom_css()
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


def handle_chat_input(top_k):
    """Handle new chat input from user."""
    if query := st.chat_input("Ask me about apple(s), BMS or FRAs..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})

        # Display user message
        with st.chat_message("user"):
            st.markdown(query)

        # Check if we should search or respond directly
        should_search, direct_response = should_search_index(query)

        if not should_search:
            # Handle non-search queries (greetings, about, etc.)
            with st.chat_message("assistant", avatar="🦍"):
                st.markdown(direct_response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": direct_response
                })
        else:
            # Perform search and generate response
            handle_search_query(query, top_k)


def handle_search_query(query, top_k):
    """Handle search queries with federated search."""
    with st.chat_message("assistant", avatar="🦍"):
        with st.spinner("Searching across indexes and analysing document dates..."):
            try:
                results, answer, publication_date_info, score_too_low = perform_federated_search(
                    query, top_k)

                # Store results in session state
                st.session_state.last_results = results

                if not results:
                    response = "I couldn't find any relevant information in our knowledge bases. Regan has told me to say I don't know."
                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                elif score_too_low:
                    # Display the low-score message
                    st.markdown(answer)
                    display_low_score_warning()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "results": results,
                        "score_too_low": True
                    })
                else:
                    if answer:
                        st.markdown(answer)

                        # Display publication date info prominently
                        if publication_date_info:
                            display_publication_date_info(
                                publication_date_info)

                        # Store message with results and publication date info
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "results": results,
                            "publication_date_info": publication_date_info
                        })
                    else:
                        # If no answer generation, show results directly
                        display_direct_results(results, publication_date_info)

            except Exception as e:
                error_msg = f"Sorry, I encountered an error while searching: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })


def display_direct_results(results, publication_date_info):
    """Display search results directly when no LLM answer is generated."""
    response = f"I found {len(results)} relevant results:"
    st.markdown(response)

    for i, result in enumerate(results, 1):
        if i == 1:
            st.markdown('<div class="top-result-highlight">🥇 <strong>TOP RESULT</strong></div>',
                        unsafe_allow_html=True)

        st.markdown(
            f"**{i}. Score:** {result.get('score', 0):.3f}  \n"
            f"_Document:_ `{result.get('key', 'Unknown')}`  •  _Index:_ `{result.get('index', '?')}`  •  _Namespace:_ `{result.get('namespace', '__default__')}`"
        )
        snippet = result.get("text") or "_(no text in metadata)_"
        st.write(snippet[:500] + "..." if len(snippet) > 500 else snippet)
        st.caption(f"ID: {result.get('id') or '—'}")
        if i < len(results):
            st.markdown("---")

    # Display publication date info for search results
    if publication_date_info:
        display_publication_date_info(publication_date_info)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "results": results,
        "publication_date_info": publication_date_info
    })


def display_last_results():
    """Display last search results in expandable section."""
    if "last_results" in st.session_state and st.session_state.last_results:
        with st.expander(f"📚 Last Search: {len(st.session_state.last_results)} results", expanded=False):
            for i, result in enumerate(st.session_state.last_results, 1):
                if i == 1:
                    st.markdown('<div class="top-result-highlight">🥇 <strong>TOP RESULT</strong></div>',
                                unsafe_allow_html=True)

                st.markdown(
                    f"**{i}. Score:** {result.get('score', 0):.3f}  \n"
                    f"_Document:_ `{result.get('key', 'Unknown')}`  •  _Index:_ `{result.get('index', '?')}`"
                )
                snippet = result.get("text") or "_(no text in metadata)_"
                st.write(snippet[:300] + "..." if len(snippet)
                         > 300 else snippet)
                if i < len(st.session_state.last_results):
                    st.markdown("---")


if __name__ == "__main__":
    main()
