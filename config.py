#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration settings for AskAlfred chatbot.
"""

import os
import streamlit as st

# Try to load a local .env if python-dotenv is available; otherwise, ignore
try:
    from dotenv import load_dotenv  # optional in Streamlit Cloud
    load_dotenv()
except Exception:
    pass

# Streamlit Cloud secrets fallback
try:
    if "PINECONE_API_KEY" not in os.environ and "PINECONE_API_KEY" in st.secrets:
        os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
    if "OPENAI_API_KEY" not in os.environ and "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

# ----- Config Constants -----
SPECIAL_INFERENCE_INDEX = "apples"
SPECIAL_INFERENCE_MODEL = "llama-text-embed-v2"
DEFAULT_NAMESPACE = "__default__"

TARGET_INDEXES = ["apples", "bms"]  # federated search targets
SEARCH_ALL_NAMESPACES = True

DEFAULT_EMBED_MODEL = os.getenv(
    "DEFAULT_EMBED_MODEL", "text-embedding-3-small")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4o-mini")
DIMENSION = 1536  # for text-embedding-3-small

# Minimum score threshold for responses
MIN_SCORE_THRESHOLD = 0.3
