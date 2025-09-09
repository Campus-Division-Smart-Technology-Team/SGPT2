#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Q&A over multiple Pinecone indexes (federated search) with chat interface.
- Enhanced with intelligent query classification (no search for greetings/about queries)
- Improved publication date detection by searching all documents from the same source
- Users do NOT select index/namespace. App searches both "apples" and "bms".
- "apples" index is forced to use Pinecone server-side inference (as before).
- Other indexes try server-side inference first, then fall back to vector search.
- Minimum score threshold of 0.3 for responses

Env:
- PINECONE_API_KEY
- OPENAI_API_KEY
- ANSWER_MODEL (default: "gpt-4o-mini")
- DEFAULT_EMBED_MODEL (default: "text-embedding-3-small")
"""

from typing import Any, Dict, List, Optional, Tuple
from heapq import nlargest
import os
from datetime import datetime
import re
import random
import logging

import streamlit as st
from pinecone import Pinecone
from openai import OpenAI

# ---------- App & env ----------
st.set_page_config(
    page_title="University of Bristol | Streamlit App",
    page_icon="https://www.bristol.ac.uk/assets/responsive-web-project/2.6.9/images/logos/uob-logo.svg",
    layout="wide",
)

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

# ----- Config -----
SPECIAL_INFERENCE_INDEX = "apples"
SPECIAL_INFERENCE_MODEL = "llama-text-embed-v2"
DEFAULT_NAMESPACE = "__default__"

TARGET_INDEXES = ["apples", "bms"]  # federated search targets
SEARCH_ALL_NAMESPACES = True

DEFAULT_EMBED_MODEL = os.getenv("DEFAULT_EMBED_MODEL", "text-embedding-3-small")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4o-mini")
DIMENSION = 1536  # for text-embedding-3-small

# Minimum score threshold for responses
MIN_SCORE_THRESHOLD = 0.3

# Setup logging
logging.basicConfig(level=logging.INFO)


# ---------- Clients ----------
def get_oai() -> OpenAI:
    return OpenAI()


def get_pc() -> Pinecone:
    return Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))


pc = get_pc()
oai = get_oai()


# ---------- Query Classification ----------
class QueryClassifier:
    """Classify user queries to determine if they need index search or direct response."""

    # Patterns that don't require search
    GREETING_PATTERNS = [
        r'^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening)|howdy)[\s!.,]*$',
        r'^(hi|hello|hey)\s+alfred[\s!.,]*$',
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
        r'^(thanks\s+and\s+bye|bye\s+for\s+now)[\s!.,]*$'
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
            "Hello! I'm Alfred 🦍, your helpful assistant at the University of Bristol. I can help you find information about:\n\n• 🍎 Apples (the fruit) and Apple Inc.\n• 🏢 Building Management Systems (BMS)\n\nWhat would you like to know today?",
            "Hi there! I'm Alfred, ready to help you search through our knowledge bases. Feel free to ask me about apples or BMS systems. How can I assist you?",
            "Hello! Alfred here, your University of Bristol assistant. I have access to information about apples and building management systems. What can I help you with?"
        ]
        return random.choice(greetings)

    @staticmethod
    def get_about_response() -> str:
        """Return information about Alfred."""
        return """I'm Alfred 🦍, a specialised assistant for the University of Bristol's Smart Technology team.

**What I can do:**
• Search and retrieve information from our knowledge bases
• Answer questions about apples (both the fruit and Apple Inc.)
• Provide information about Building Management Systems (BMS) at the university
• Tell you when documents were last updated or published

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
    Determine if a query requires index search or can be answered directly.

    Returns:
        (should_search, direct_response)
    """
    query_type, suggested_response = QueryClassifier.classify_query(query)

    if query_type == 'search':
        return True, None
    else:
        return False, suggested_response


# ---------- Pinecone helpers ----------
def list_index_names() -> List[str]:
    try:
        idxs = pc.list_indexes()
    except Exception:
        return []
    if hasattr(idxs, "names"):
        return list(idxs.names())
    if isinstance(idxs, dict) and "indexes" in idxs:
        return [i["name"] for i in idxs["indexes"]]
    return list(idxs) if isinstance(idxs, (list, tuple)) else []


def open_index(name: str):
    return pc.Index(name)


def list_namespaces_for_index(idx) -> List[str]:
    """Return available namespaces for an index; '__default__' represents default."""
    try:
        stats = idx.describe_index_stats()
        ns_dict = (stats or {}).get("namespaces") or {}
        names = list(ns_dict.keys())
        names = [n if isinstance(n, str) else DEFAULT_NAMESPACE for n in names]
        if DEFAULT_NAMESPACE not in names and "" not in names:
            names.append(DEFAULT_NAMESPACE)
        names = [DEFAULT_NAMESPACE if n == "" else n for n in names]
        names = sorted(list(set(names)), key=lambda n: (n != DEFAULT_NAMESPACE, n))
        return names
    except Exception:
        return [DEFAULT_NAMESPACE]


# ---------- Search utilities ----------
def embed_texts(texts: List[str], model: str) -> List[List[float]]:
    res = oai.embeddings.create(model=model, input=texts)
    return [d.embedding for d in res.data]


def vector_query(idx, namespace: str, query: str, k: int, embed_model: str) -> Dict[str, Any]:
    vec = embed_texts([query], embed_model)[0]
    return idx.query(vector=vec, top_k=k, namespace=namespace, include_metadata=True)


def try_inference_search(idx, ns: str, q: str, k: int, model_name: Optional[str] = None):
    """
    Prefer Pinecone Index.search (server-side inference). If unavailable, embed on
    server via pc.inference and then Index.query.
    """
    if hasattr(idx, "search"):
        try:
            return idx.search(namespace=ns, inputs={"text": q}, top_k=k, include_metadata=True)
        except TypeError:
            pass
        except Exception:
            pass
        return idx.search(namespace=ns, query={"inputs": {"text": q}, "top_k": k})

    # Fallback: server-side embeddings then query
    try:
        embs = pc.inference.embed(
            model=(model_name or SPECIAL_INFERENCE_MODEL),
            inputs=[q],
            parameters={"input_type": "query", "truncate": "END"}
        )
        first = embs.data[0]
        vec = first.get("values") if isinstance(first, dict) else getattr(first, "values", None)
        if vec is None:
            vec = first.get("embedding") if isinstance(first, dict) else getattr(first, "embedding", None)
        if vec is None:
            raise RuntimeError("Unexpected embeddings response shape; no vector values found")
    except Exception as e:
        raise RuntimeError(f"Server-side embedding failed: {e}")

    return idx.query(vector=vec, top_k=k, namespace=ns, include_metadata=True)


def _as_dict(obj: Any) -> Dict[str, Any]:
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            pass
    return obj if isinstance(obj, dict) else {}


def normalize_matches(raw: Any) -> List[Dict[str, Any]]:
    """Normalise Pinecone results from either `matches` or `result.hits` shapes."""
    data = _as_dict(raw)

    if isinstance(data, dict) and isinstance(data.get("matches"), list):
        out: List[Dict[str, Any]] = []
        for m in data["matches"]:
            md = m.get("metadata") if isinstance(m, dict) else {}
            out.append({
                "id": m.get("id"),
                "score": m.get("score"),
                "metadata": md or {},
                "text": (md or {}).get("text") or (md or {}).get("content") or (md or {}).get("chunk") or (
                        md or {}).get("body") or "",
                "source": (md or {}).get("source") or (md or {}).get("url") or (md or {}).get("doc") or "",
                "key": (md or {}).get("key") or "",  # Extract key from metadata
            })
        return out

    hits = (data.get("result") or {}).get("hits") if isinstance(data, dict) else []
    if isinstance(hits, list) and hits:
        out = []
        for h in hits:
            fields = h.get("fields") or h.get("metadata") or {}
            text_val = fields.get("text") or fields.get("content") or fields.get("chunk") or fields.get("body") or ""
            out.append({
                "id": h.get("_id"),
                "score": h.get("_score"),
                "metadata": fields,
                "text": text_val,
                "source": fields.get("source") or fields.get("url") or fields.get("doc") or "",
                "key": fields.get("key") or "",  # Extract key from metadata
            })
        return out

    return []


# ---------- Enhanced Publication Date Search ----------
def parse_date_string(date_str: str) -> datetime:
    """Parse various date formats and return a datetime object."""
    if not date_str or date_str == "publication date unknown":
        return datetime.min

    # Common date formats to try (including dot format)
    formats = [
        "%d.%m.%Y",     # 12.06.2025 (dot format)
        "%Y.%m.%d",     # 2025.06.12 (dot format)
        "%Y-%m-%d",     # 2025-07-28
        "%d %B %Y",     # 28 July 2025
        "%B %d, %Y",    # July 28, 2025
        "%d/%m/%Y",     # 28/07/2025
        "%m/%d/%Y",     # 07/28/2025
        "%Y",           # 2025
        "%B %Y",        # July 2025
        "%d-%m-%Y",     # 28-07-2025
        "%Y/%m/%d",     # 2025/07/28
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue

    # If no format matches, return minimum date
    return datetime.min


def format_display_date(date_obj: datetime) -> str:
    """Format datetime object for display."""
    if date_obj.day == 1 and date_obj.month == 1:
        # Year only
        return date_obj.strftime("%Y")
    elif date_obj.day == 1:
        # Month and year
        return date_obj.strftime("%B %Y")
    else:
        # Full date
        return date_obj.strftime("%d %B %Y")


def search_source_for_latest_date(idx, key_value: str, namespace: str = DEFAULT_NAMESPACE) -> Tuple[
    Optional[str], List[Dict[str, Any]]]:
    """
    Search for all documents from the same key and determine the latest publication/review date.

    Returns:
        Tuple of (latest_date_string, all_matching_documents)
    """
    try:
        # Query for documents with the same key
        source_query = f"key:{key_value} publication date review date updated revised"

        # Try to get many results to find all documents from this source
        try:
            raw = try_inference_search(idx, namespace, source_query, k=20, model_name=None)
        except:
            # Fallback to vector search
            raw = vector_query(idx, namespace, source_query, 20, DEFAULT_EMBED_MODEL)

        results = normalize_matches(raw)

        # Filter results to only those from the same key
        matching_docs = [r for r in results if r.get("key") == key_value]

        if not matching_docs:
            # If no exact matches, try without filtering
            matching_docs = results[:10]  # Take top 10 results

        # Extract all possible dates from matching documents
        all_dates = []
        date_patterns = [
            # Dot format patterns
            (r'\b(\d{1,2}\.\d{1,2}\.\d{4})\b', 'dot_dmy'),
            (r'\b(\d{4}\.\d{1,2}\.\d{1,2})\b', 'dot_ymd'),
            # Context-aware patterns
            (r'\b(?:published|updated|revised|reviewed)[\s:]+(\d{1,2}[/.]\d{1,2}[/.]\d{4})\b', 'full'),
            (r'\b(?:published|updated|revised|reviewed)[\s:]+(\d{4}[/.-]\d{1,2}[/.-]\d{1,2})\b', 'iso'),
            (r'\b(?:published|updated|revised|reviewed)[\s:]+(?:in\s+)?(\d{4})\b', 'year'),
            (r'\b(?:Publication Date|Review Date|Last Updated|Last Revised)[\s:]+([^,\n]+)', 'labeled'),
            (r'\b(?:©|Copyright)\s+(\d{4})\b', 'copyright'),
            # Generic patterns
            (r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b', 'generic_full'),
            (r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b', 'generic_iso'),
        ]

        for doc in matching_docs:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})

            # Check metadata fields EXCEPT publication_date (as it's misleading)
            for date_field in ["review_date", "updated", "revised", "date", "last_modified"]:
                if date_field in metadata:
                    date_val = metadata[date_field]
                    if date_val and date_val != "publication date unknown":
                        parsed = parse_date_string(str(date_val))
                        if parsed != datetime.min:
                            all_dates.append((parsed, str(date_val), doc.get("id")))

            # Check text content with context-aware patterns
            for pattern, pattern_type in date_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    parsed = parse_date_string(match)
                    if parsed != datetime.min:
                        # Validate date is reasonable (not in future, not too old)
                        current_date = datetime.now()
                        if parsed <= current_date and (current_date - parsed).days < 20 * 365:
                            all_dates.append((parsed, match, doc.get("id")))

        # Find the latest date
        if all_dates:
            all_dates.sort(key=lambda x: x[0], reverse=True)
            latest_date_obj, latest_date_str, doc_id = all_dates[0]
            return latest_date_str, matching_docs

        return None, matching_docs

    except Exception as e:
        logging.error(f"Error searching for source dates: {e}")
        return None, []


# ---------- Answering ----------
def build_context(snippets: List[str], max_chars: int = 6000) -> str:
    blob = ("\n\n".join(s.strip() for s in snippets if s.strip()))
    return blob if len(blob) <= max_chars else blob[:max_chars]


def enhanced_answer_with_source_date(question: str, top_result: Dict[str, Any], all_results: List[Dict[str, Any]]) -> \
Tuple[str, str]:
    """
    Generate an answer that includes information about the source's latest publication date.
    Returns: (answer, publication_date_info)
    """
    # Get the key from top result
    key_value = top_result.get("key", "")
    idx_name = top_result.get("index", "")
    source_value = top_result.get("source", "")

    latest_date = None
    source_doc_count = 1

    if key_value and idx_name:
        idx = open_index(idx_name)
        latest_date, source_docs = search_source_for_latest_date(
            idx, key_value, top_result.get("namespace", DEFAULT_NAMESPACE)
        )
        source_doc_count = len([d for d in source_docs if d.get("key") == key_value])

    # Format date information
    if latest_date:
        parsed = parse_date_string(latest_date)
        display_date = format_display_date(parsed)
        date_context = f"The document '{key_value}' was last updated/reviewed on {display_date}."
        publication_info = f"📅 Document last updated: **{display_date}** (from {key_value}, searched {source_doc_count} related chunks)"
    else:
        date_context = f"The publication date for document '{key_value}' could not be determined."
        publication_info = f"📅 **Publication date unknown** -- {source_value} --)"

    # Build context from all results
    snippets = [r.get("text", "") for r in all_results if r.get("text")]
    context = build_context(snippets)

    prompt = f"""Your name is Alfred, a helpful assistant at the University of Bristol working in the Smart Technology team.

Answer the user's question using ONLY the context below. If the answer cannot be found in the context, tell the user that Regan has told you to say you don't know.

IMPORTANT: Always end your response with information about when the document was last updated or published.

Question: {question}

Context: {context}

Document Information: {date_context}

Top Result Details:
- Document: {key_value}
- Score: {top_result.get('score', 'Unknown'):.3f}
- Index: {idx_name}
"""

    chat = oai.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system",
             "content": "You are Alfred, a helpful assistant. Always include document date information in your responses."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    answer = chat.choices[0].message.content.strip()
    return answer, publication_info


# ---------- Federated search helpers ----------
def _namespaces_to_search(idx):
    if not SEARCH_ALL_NAMESPACES:
        return [DEFAULT_NAMESPACE]
    try:
        return list_namespaces_for_index(idx)
    except Exception:
        return [DEFAULT_NAMESPACE]


def search_one_index(idx_name: str, question: str, k: int, embed_model: Optional[str]) -> List[Dict[str, Any]]:
    """Query a single index across its namespaces; attach provenance."""
    idx = open_index(idx_name)
    hits: List[Dict[str, Any]] = []

    force_inference = (idx_name == SPECIAL_INFERENCE_INDEX)
    namespaces = _namespaces_to_search(idx)

    for ns in namespaces:
        try:
            if force_inference:
                raw = try_inference_search(idx, ns, question, k, model_name=SPECIAL_INFERENCE_MODEL)
                mode_used = "server-side (inference)"
            else:
                try:
                    raw = try_inference_search(idx, ns, question, k, model_name=None)
                    mode_used = "server-side (inference)"
                except Exception:
                    raw = vector_query(idx, ns, question, k, embed_model or DEFAULT_EMBED_MODEL)
                    mode_used = "client-side (vector)"
            norm = normalize_matches(raw)
            for m in norm:
                m["index"] = idx_name
                m["namespace"] = ns
                m["_mode"] = mode_used
            hits.extend(norm)
        except Exception:
            # non-fatal per-namespace failure
            continue

    return hits


def perform_federated_search(query: str, top_k: int) -> tuple[List[Dict[str, Any]], str, str, bool]:
    """
    Perform federated search and return results, answer, publication date info, and low score flag.
    Returns: (results, answer, publication_date_info, score_too_low)
    """
    all_hits: List[Dict[str, Any]] = []

    for idx_name in TARGET_INDEXES:
        all_hits.extend(search_one_index(idx_name, query, top_k, embed_model=None))

    # Merge & keep a global top_k
    top_hits = nlargest(top_k, all_hits, key=lambda m: (m.get("score") or 0))

    answer = ""
    publication_date_info = ""
    score_too_low = False

    # Check if top score is below threshold
    if top_hits:
        top_score = top_hits[0].get("score", 0)
        if top_score < MIN_SCORE_THRESHOLD:
            score_too_low = True
            answer = f"I found some results, but they don't seem relevant enough to your question. The best matching score was {top_score:.3f} which is below the allowable threshold of {MIN_SCORE_THRESHOLD}. Regan has told me to say I don't know. Please try rephrasing your question or asking about something else."
            return top_hits, answer, publication_date_info, score_too_low

    if top_hits and st.session_state.get("generate_llm_answer", True):
        # Use enhanced answer generation with source date search
        answer, publication_date_info = enhanced_answer_with_source_date(query, top_hits[0], top_hits)
    elif top_hits:
        # If LLM answer generation is disabled, still try to get date info
        top_result = top_hits[0]
        key_value = top_result.get("key", "")
        if key_value and top_result.get("index"):
            idx = open_index(top_result.get("index"))
            latest_date, _ = search_source_for_latest_date(
                idx, key_value, top_result.get("namespace", DEFAULT_NAMESPACE)
            )
            if latest_date:
                parsed = parse_date_string(latest_date)
                display_date = format_display_date(parsed)
                publication_date_info = f"📅 Top result document last updated: **{display_date}**"
            else:
                publication_date_info = f"📅 **Publication date unknown** for top result"

    return top_hits, answer, publication_date_info, score_too_low


# ---------- UI ----------
st.markdown(
    """
    <style>
      .uob-header {
        position: relative;
        background: rgba(227, 230, 229, 0.7);
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        display: flex;
        align-items: center;
        gap: 14px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
      }
      @media (prefers-color-scheme: light) {
        .uob-header { background: rgba(171, 31, 45, 0.4); border: 1px solid rgba(0, 0, 0, 0.1); }
      }
      @media (prefers-color-scheme: dark) {
        .uob-header h1 { color: #000 !important; }
      }
      .uob-header img { height: 70px; z-index: 2;}
      .uob-header h1 {
        position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%);
        margin: 0; font-size: 2rem;
      }
      .publication-date {
        background-color: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
        padding: 8px 12px;
        margin: 8px 0;
        border-radius: 4px;
        font-size: 0.9em;
      }
      .top-result-highlight {
        background-color: rgba(40, 167, 69, 0.1);
        border-left: 4px solid #28a745;
        padding: 8px 12px;
        margin: 8px 0;
        border-radius: 4px;
        font-size: 0.9em;
      }
      .low-score-warning {
        background-color: rgba(220, 53, 69, 0.1);
        border-left: 4px solid #dc3545;
        padding: 8px 12px;
        margin: 8px 0;
        border-radius: 4px;
        font-size: 0.9em;
      }
    </style>

    <div class="uob-header">
      <picture>
        <source srcset="https://www.bristol.ac.uk/assets/responsive-web-project/2.6.9/images/logos/uob-logo.svg" media="(prefers-color-scheme: light)"/>
        <source srcset="https://www.bristol.ac.uk/assets/responsive-web-project/2.6.9/images/logos/uob-logo.svg"/>
        <img src="https://www.bristol.ac.uk/assets/responsive-web-project/2.6.9/images/logos/uob-logo.svg" alt="University of Bristol"/>
      </picture>
      <h1>🦍 Ask Alfred</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Tabs: Welcome | Info | Resources ----
tab1, tab2, tab3 = st.tabs(["Welcome", "Info", "Resources"])

with tab1:
    st.write(
        """
        #### Hi, I'm Alfred! 👋  
        You can ask me questions about:
        - 🍎 Apples (the fruit) and 💻 Apple Inc.  
        - 🏢 Building Management Systems (BMS) at the University of Bristol  

        Type your question in the chat below, and I'll search across our knowledge bases to find answers.
        """
    )

with tab2:
    st.write(
        """
        #### ⚠️ Disclaimer
        This app is experimental and should not be used for decision-making. Please note that the data used in creating the apples index was synthetically generated by ***ChatGPT 5***.  
        The chatbot is configured to say **"Regan has told me to say I don't know."** if the answer isn't in the knowledge base or relevance is too low.

        #### 🔍 Enhanced Features
        **Smart Query Classification:**
        - Instantly responds to greetings without searching
        - Provides information about Alfred's capabilities without API calls
        - Handles gratitude and farewells appropriately

        **Enhanced Date Search:**
        - Searches ALL chunks from the same document (using 'key' metadata)
        - Supports multiple date formats including dots (12.06.2025)
        - Ignores misleading 'publication_date' from index metadata
        - Uses context-aware patterns to identify publication dates
        - Validates dates to ensure they're reasonable

        **Quality Control:**
        - Minimum score threshold of 0.3 for responses
        - Won't generate answers from low-relevance results
        """
    )
    # Footer with accessibility statement
    st.markdown("""
    ---
    <footer role="contentinfo" style="margin-top: 2rem; padding: 1rem; background-color: rgba(0,0,0,0.05); border-radius: 8px;">
        <small>
        <strong>Accessibility:</strong> This application follows WCAG 2.2 AA guidelines. 
        If you encounter any accessibility issues, please contact the <strong>Smart Technology Data Team</strong>.<br>
        <strong>University of Bristol</strong> | Experimental Research Application
        </small>
    </footer>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("#### 💡 Example queries")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            """
            **Apple(s) topics:**
            - "What is Apple's flagship product?"
            - "Tell me about some Apple products"
            - "Tell me about the different types of apples"

            **General queries:**
            - "Hello Alfred"
            - "Who are you?"
            - "What can you help with?"
            """
        )
    with col2:
        st.markdown(
            """
            **BMS topics:**
            - "How does the frost protection sequence operate in the Berkeley Square and Indoor Sports Hall BMS systems?"
            - "What access levels are defined for controllers in the Retort House?"
            - "How does the Mitsubishi AC controller integrate with the Trend IQ4 BMS?"
            """
        )

# ---- Sidebar Settings ----
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Results per query", min_value=1, max_value=25, value=5)

    if "generate_llm_answer" not in st.session_state:
        st.session_state.generate_llm_answer = True

    generate_llm_answer = st.checkbox(
        "Generate AI answer from search results",
        value=st.session_state.generate_llm_answer,
        help="If disabled, you'll only see the retrieved passages."
    )
    st.session_state.generate_llm_answer = generate_llm_answer

    st.markdown("---")
    st.info(f"**Minimum Score Threshold:** {MIN_SCORE_THRESHOLD}")

    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.last_results = []
        st.rerun()

    st.markdown("---")
    with st.expander("Search Details"):
        st.write(f"**Indexes:** {', '.join(TARGET_INDEXES)}")
        st.write(
            f"**Namespaces:** {'all available' if SEARCH_ALL_NAMESPACES else DEFAULT_NAMESPACE}")
        st.caption(
            "Server-side inference is forced for 'apples'; others try inference then vector.")
        st.caption(
            "Enhanced: Smart query classification, document-level date search, and relevance threshold.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm Alfred 🦍, your helpful assistant at the University of Bristol. I can help you find information about apples and Building Management Systems. What would you like to know?"
        }
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display publication date info if it exists
        if "publication_date_info" in message and message["publication_date_info"]:
            st.markdown(f'<div class="publication-date">{message["publication_date_info"]}</div>',
                        unsafe_allow_html=True)

        # Display low score warning if applicable
        if message.get("score_too_low", False):
            st.markdown('<div class="low-score-warning">⚠️ Results below relevance threshold</div>',
                        unsafe_allow_html=True)

        # Display search results if they exist
        if "results" in message:
            with st.expander(f"📚 Search Results ({len(message['results'])} found)", expanded=False):
                for i, result in enumerate(message["results"], 1):
                    # Highlight the top result
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
                    if i < len(message["results"]):
                        st.markdown("---")

# Chat input
if query := st.chat_input("Ask me about apple(s) or BMS..."):
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
        with st.chat_message("assistant", avatar="🦍"):
            with st.spinner("Searching across indexes and analysing document dates..."):
                try:
                    results, answer, publication_date_info, score_too_low = perform_federated_search(query, top_k)

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
                        st.markdown('<div class="low-score-warning">⚠️ Results below relevance threshold</div>',
                                    unsafe_allow_html=True)

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
                                st.markdown(f'<div class="publication-date">{publication_date_info}</div>',
                                            unsafe_allow_html=True)

                            # Store message with results and publication date info
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "results": results,
                                "publication_date_info": publication_date_info
                            })
                        else:
                            # If no answer generation, show results directly
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
                                st.markdown(f'<div class="publication-date">{publication_date_info}</div>',
                                            unsafe_allow_html=True)

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response,
                                "results": results,
                                "publication_date_info": publication_date_info
                            })

                except Exception as e:
                    error_msg = f"Sorry, I encountered an error while searching: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

# Display last results in expandable section if they exist
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
            st.write(snippet[:300] + "..." if len(snippet) > 300 else snippet)
            if i < len(st.session_state.last_results):
                st.markdown("---")
