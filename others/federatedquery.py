#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Q&A over multiple Pinecone indexes (federated search).
- Users do NOT select index/namespace. App searches both "apples" and "bms".
- "apples" index is forced to use Pinecone server-side inference (as before).
- Other indexes try server-side inference first, then fall back to vector search.
- Retrieved context is merged and (optionally) used to generate a single answer.

Env:
- PINECONE_API_KEY
- OPENAI_API_KEY
- ANSWER_MODEL (default: "gpt-4o-mini")
- DEFAULT_EMBED_MODEL (default: "text-embedding-3-small")
"""

from typing import Any, Dict, List, Optional
from heapq import nlargest
import os

import streamlit as st
from pinecone import Pinecone
from openai import OpenAI

# ---------- App & env ----------
st.set_page_config(
    page_title="University of Bristol | Streamlit App",
    page_icon="https://cdn.brandfetch.io/idWwwm9Vvi/w/820/h/237/theme/dark/logo.png?c=1dxbfHSJFAPEGdCLU4o5B",
    layout="wide",
)

# Try to load a local .env if python-dotenv is available; otherwise ignore
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
# force Pinecone inference for this index
SPECIAL_INFERENCE_INDEX = "apples"
# server-side model config on index
SPECIAL_INFERENCE_MODEL = "llama-text-embed-v2"
DEFAULT_NAMESPACE = "__default__"

TARGET_INDEXES = ["apples", "bms"]                 # federated search targets
# True -> search every namespace per index
SEARCH_ALL_NAMESPACES = True

DEFAULT_EMBED_MODEL = os.getenv(
    "DEFAULT_EMBED_MODEL", "text-embedding-3-small")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4o-mini")

# ---------- Clients ----------


def get_oai() -> OpenAI:
    return OpenAI()


def get_pc() -> Pinecone:
    return Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))


pc = get_pc()
oai = get_oai()

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
        names = sorted(list(set(names)), key=lambda n: (
            n != DEFAULT_NAMESPACE, n))
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
    server via pc.inference and then Index.query. (Same behavior you used before.)
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
        vec = first.get("values") if isinstance(
            first, dict) else getattr(first, "values", None)
        if vec is None:
            vec = first.get("embedding") if isinstance(
                first, dict) else getattr(first, "embedding", None)
        if vec is None:
            raise RuntimeError(
                "Unexpected embeddings response shape; no vector values found")
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
                "text": (md or {}).get("text") or (md or {}).get("content") or (md or {}).get("chunk") or (md or {}).get("body") or "",
                "source": (md or {}).get("source") or (md or {}).get("url") or (md or {}).get("doc") or "",
            })
        return out

    hits = (data.get("result") or {}).get(
        "hits") if isinstance(data, dict) else []
    if isinstance(hits, list) and hits:
        out = []
        for h in hits:
            fields = h.get("fields") or h.get("metadata") or {}
            text_val = fields.get("text") or fields.get(
                "content") or fields.get("chunk") or fields.get("body") or ""
            out.append({
                "id": h.get("_id"),
                "score": h.get("_score"),
                "metadata": fields,
                "text": text_val,
                "source": fields.get("source") or fields.get("url") or fields.get("doc") or "",
            })
        return out

    return []

# ---------- Answering ----------


def build_context(snippets: List[str], max_chars: int = 6000) -> str:
    blob = ("\n\n".join(s.strip() for s in snippets if s.strip()))
    return blob if len(blob) <= max_chars else blob[:max_chars]


def answer_question(question: str, context: str) -> str:
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer cannot be found in the context, tell the user that "Regan" has told you to say you don't know.

# Question
{question}

# Context
{context}
"""
    chat = oai.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": "You answer clearly and concisely."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return chat.choices[0].message.content.strip()

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
                raw = try_inference_search(
                    idx, ns, question, k, model_name=SPECIAL_INFERENCE_MODEL)
                mode_used = "server-side (inference)"
            else:
                try:
                    raw = try_inference_search(
                        idx, ns, question, k, model_name=None)
                    mode_used = "server-side (inference)"
                except Exception:
                    raw = vector_query(idx, ns, question, k,
                                       embed_model or DEFAULT_EMBED_MODEL)
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
    </style>

    <div class="uob-header">
      <picture>
        <source srcset="https://www.bristol.ac.uk/assets/responsive-web-project/2.6.9/images/logos/uob-logo.svg" media="(prefers-color-scheme: light)"/>
        <source srcset="https://www.bristol.ac.uk/assets/responsive-web-project/2.6.9/images/logos/uob-logo.svg"/>
        <img src="https://www.bristol.ac.uk/assets/responsive-web-project/2.6.9/images/logos/uob-logo.svg" alt="University of Bristol"/>
      </picture>
      <h1>Apple(s) &amp; BMS</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Tabs: Welcome | Disclaimer | Examples ----
tab1, tab2, tab3 = st.tabs(["Welcome", "Disclaimer", "Examples"])

with tab1:
    st.write(
        """
        #### Hello! 👋  
        Ask questions about:
        - 🍎 Apples (the fruit) and 💻 Apple Inc.  
        - 🏢 Building Management Systems (BMS) at the University of Bristol  
        """
    )

with tab2:
    st.write(
        """
        #### ⚠️ Disclaimer
        This app is experimental and should not be used for decision-making. Please note that the data used in creating the apples index was synthetically generated by ***ChatGPT 5***.  
        The chatbot is configured to say **"Regan has told me to say I don't know."** if the answer isn't in the knowledge base.
        """
    )

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
            """
        )
    with col2:
        st.markdown(
            """
            **BMS topics:**
            - "How does the frost protection sequence operate in the Berkeley Square and Indoor Sports Hall BMS systems?"
            - "What access levels are defined for controllers in the Retort House and Dentistry BMS manuals?"
            - "How does the Mitsubishi AC controller integrate with the Trend IQ4 BMS?"
            """
        )

# ---- Sidebar (no index/namespace pickers) ----
with st.sidebar:
    st.header("Connection")
    top_k = st.slider("Top K", min_value=1, max_value=25, value=5)
    generate_llm_answer = st.checkbox(
        "Generate an OpenAI answer from retrieved results",
        value=True,
        help="If off, you'll only see the retrieved passages."
    )
    st.markdown("---")
    with st.expander("Search details"):
        st.write(f"**Indexes:** {', '.join(TARGET_INDEXES)}")
        st.write(
            f"**Namespaces:** {'all available' if SEARCH_ALL_NAMESPACES else DEFAULT_NAMESPACE}")
        st.caption(
            "Server-side inference is forced for 'apples'; others try inference then vector.")

st.markdown("---")

# ---- Input widget ----
query = st.text_input(
    "To get started, type your question below.", placeholder="Ask me about apple(s) or BMS"
)
col_search, col_space, col_space2, col_clear = st.columns([1, 2, 2, 1])
with col_search:
    go = st.button("🔍 Search")
with col_clear:
    if st.button("🔄 Clear"):
        st.rerun()

# ---- Federated Query/Results flow ----
if go and query.strip():
    with st.spinner("Searching across indexes…", show_time=True):
        all_hits: List[Dict[str, Any]] = []
        for idx_name in TARGET_INDEXES:
            all_hits.extend(search_one_index(
                idx_name, query, top_k, embed_model=None))

        # Merge & keep a global top_k (note: scores may not be strictly comparable)
        top_hits = nlargest(
            top_k, all_hits, key=lambda m: (m.get("score") or 0))

    if not top_hits:
        st.info("No results.")
        st.stop()

    # Optional LLM answer built from combined snippets
    if generate_llm_answer:
        snippets = [m.get("text", "") for m in top_hits if m.get("text")]
        context = build_context(snippets)
        try:
            answer = answer_question(query, context)
        except Exception as e:
            answer = f"(Answer generation failed) {e}"

        st.subheader("Answer")
        st.write(answer)

    st.subheader("Results")
    for i, m in enumerate(top_hits, start=1):
        with st.container():
            st.markdown(
                f"**{i}. Score:** {m.get('score')}  \n"
                f"_Index:_ `{m.get('index','?')}`  •  _Namespace:_ `{m.get('namespace','__default__')}`  •  _Mode:_ {m.get('_mode','')}"
            )
            snippet = m.get("text") or "_(no text in metadata)_"
            st.write(snippet)
            src = m.get("source") or "—"
            st.caption(f"ID: {m.get('id') or '—'} • Source: {src}")
            st.markdown("---")

else:
    st.info("👆 Enter a question, press **🔍 Search** to get started, and use **🔄 clear** to reset.")

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
