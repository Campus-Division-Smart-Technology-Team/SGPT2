#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Q&A over a Pinecone index.

Behavior:
- When the selected index is exactly "llama-text-embed-v2-index", use Pinecone
  server-side inference with the attached model (llama-text-embed-v2) to search
  and ONLY display the search results (no LLM-generated answer).
- Other indexes behave as usual: you can use server-side inference when available
  or client-side vector search; the retrieved context is then used to generate an
  answer with an LLM (configurable by env).

Notes for Pinecone >=7:
- `Index.search` takes top-level kwargs like `inputs={"text": q}`, `top_k`,
  `filter`, `include_metadata`. Do not include `model` there; model comes from
  the index's attached inference config.
- Some responses return `{"matches": [...]}`. Others return
  `{"result": {"hits": [...]}}`. This app normalises both.

Environment variables:
- PINECONE_API_KEY
- OPENAI_API_KEY (for answer generation or local embeddings)
- ANSWER_MODEL (default: "gpt-4o-mini")
- DEFAULT_EMBED_MODEL (default: "text-embedding-3-small")
"""

from typing import Any, Dict, List, Optional
import os

import streamlit as st
from pinecone import Pinecone
from openai import OpenAI

# ---------- App & env ----------
st.set_page_config(page_title="Apples & BMS", page_icon="🔎", layout="wide")

# Try to load a local .env if python-dotenv is available; otherwise ignore
try:
    from dotenv import load_dotenv  # optional in Streamlit Cloud
    load_dotenv()
except Exception:
    pass

# Streamlit Cloud secrets fallback (so the app works without python-dotenv)
try:
    if "PINECONE_API_KEY" not in os.environ and "PINECONE_API_KEY" in st.secrets:
        os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
    if "OPENAI_API_KEY" not in os.environ and "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    # st.secrets may not exist locally; ignore
    pass

SPECIAL_INFERENCE_INDEX = "llama-text-embed-v2-index"
SPECIAL_INFERENCE_MODEL = "llama-text-embed-v2"

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
    """Return available namespaces for an index ('' means default namespace).
    Tries `describe_index_stats()` and falls back gracefully.
    """
    try:
        stats = idx.describe_index_stats()
        ns_dict = (stats or {}).get("namespaces") or {}
        names = list(ns_dict.keys())
        # Some deployments report None / empty keys inconsistently
        names = [n if isinstance(n, str) else "" for n in names]
        # Ensure at least default namespace exists
        if "" not in names:
            names.append("")
        # Sort with non-empty first, then default at end for clarity
        names = sorted(names, key=lambda n: (n == "", n))
        return names
    except Exception:
        # If stats not available, just expose default namespace
        return [""]


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

# ---------- Search utilities ----------


def embed_texts(texts: List[str], model: str) -> List[List[float]]:
    res = oai.embeddings.create(model=model, input=texts)
    return [d.embedding for d in res.data]


def vector_query(idx, namespace: str, query: str, k: int, embed_model: str) -> Dict[str, Any]:
    vec = embed_texts([query], embed_model)[0]
    return idx.query(vector=vec, top_k=k, namespace=namespace or "", include_metadata=True)


def try_inference_search(idx, ns: str, q: str, k: int, model_name: Optional[str] = None):
    """
    Use integrated text search when available (Index.search). That endpoint accepts
    only `text` or `inputs: {text: ...}` and `top_k` as top-level kwargs. It does
    NOT accept `model`—the model is taken from the index's inference config.

    If Index.search is not available (older SDK), fall back to server-side
    embeddings (pc.inference.embed) and then query with the vector via Index.query.
    """
    # New SDK path: integrated text search (preferred)
    if hasattr(idx, "search"):
        # 1) Preferred: top-level kwargs
        try:
            return idx.search(namespace=ns or "", inputs={"text": q}, top_k=k, include_metadata=True)
        except TypeError:
            # Some variants only accept everything in a single dict under `query`
            pass
        except Exception:
            # Try alternate shape below
            pass
        # 2) Alternate: single `query` dict
        try:
            return idx.search(namespace=ns or "", query={"inputs": {"text": q}, "top_k": k})
        except Exception as e:
            raise

    # Compatibility path for older SDKs without Index.search
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

    return idx.query(vector=vec, top_k=k, namespace=ns or "", include_metadata=True)


def _as_dict(obj: Any) -> Dict[str, Any]:
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            pass
    return obj if isinstance(obj, dict) else {}


def normalize_matches(raw: Any) -> List[Dict[str, Any]]:
    """ Normalise Pinecone results from either `matches` or `result.hits` shapes."""
    data = _as_dict(raw)
    matches = []

    if isinstance(data, dict) and isinstance(data.get("matches"), list):
        # Standard vector query shape
        matches = data["matches"]
        out: List[Dict[str, Any]] = []
        for m in matches:
            md = m.get("metadata") if isinstance(m, dict) else {}
            out.append({
                "id": m.get("id"),
                "score": m.get("score"),
                "metadata": md or {},
                "text": (md or {}).get("text") or (md or {}).get("content") or (md or {}).get("chunk") or (md or {}).get("body") or "",
                "source": (md or {}).get("source") or (md or {}).get("url") or (md or {}).get("doc") or "",
            })
        return out

    # Inference search sometimes returns {"result": {"hits": [...]}}
    hits = (data.get("result") or {}).get(
        "hits") if isinstance(data, dict) else []
    if isinstance(hits, list) and hits:
        out: List[Dict[str, Any]] = []
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
    blob = (
        "\n\n".join(s.strip() for s in snippets if s.strip())
    )
    return blob if len(blob) <= max_chars else blob[:max_chars]


def answer_question(question: str, context: str) -> str:
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer cannot be found in the context, say you don't know.

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


# ---------- UI ----------
st.title("🔎 Apple(s) & BMS")

with st.sidebar:
    st.header("Connection")
    names = list_index_names()
    if names:
        index_name = st.selectbox("Index", options=names, index=0)
    else:
        st.warning("No Pinecone indexes found for this API key.")
        index_name = None

    # --- Namespace selector (auto-populate from index stats) ---
    if index_name:
        idx_for_ns = open_index(index_name)
        ns_options = list_namespaces_for_index(idx_for_ns)
        # Choose a sensible default: first non-empty if any, else ''
        default_ns = next((n for n in ns_options if n), "")
        display_labels = ["(default)" if n == "" else n for n in ns_options]
        try:
            default_ix = display_labels.index(
                "(default)") if default_ns == "" else display_labels.index(default_ns)
        except ValueError:
            default_ix = 0
        selected_label = st.selectbox(
            "Namespace", options=display_labels, index=default_ix, help="From index stats")
        namespace = "" if selected_label == "(default)" else selected_label
    else:
        # Fallback in case there are no indexes yet
        namespace = st.text_input("Namespace", value="example-namespace")

    top_k = st.slider("Top K", min_value=1, max_value=25, value=5)

    force_inference_for_special = (index_name == SPECIAL_INFERENCE_INDEX)

    if force_inference_for_special:
        st.selectbox(
            "Query mode",
            options=["Inference (server-side)"],
            index=0,
            disabled=True,
            help="This index uses Pinecone server-side inference (llama-text-embed-v2).",
            key="qm_locked",
        )
        # Optional: allow user to also generate an OpenAI answer from retrieved results.
        generate_llm_answer_for_special = st.checkbox(
            "Also generate an OpenAI answer (uses retrieved results as context)", value=False,
            help="Safe to enable: retrieval still uses Pinecone; OpenAI only generates the final text.")
        query_mode = "Inference (server-side)"
        EMBED_MODEL = None
    else:
        query_mode = st.selectbox(
            "Query mode",
            options=["Auto", "Inference (server-side)",
                     "Vector (client-side)"],
            index=0,
            help="Auto tries server-side inference first, then falls back to vector query.",
        )
        EMBED_MODEL = st.text_input(
            "Embedding model (Vector mode)",
            value=DEFAULT_EMBED_MODEL,
            help="Used only when 'Vector (client-side)' is chosen or Auto falls back.",
        )

    st.markdown("---")
    with st.expander("Index details"):
        st.write(f"**Index:** {index_name or '—'}")
        if force_inference_for_special:
            st.write(f"**Forced inference model:** {SPECIAL_INFERENCE_MODEL}")
        st.caption(
            "Server-side inference runs inside Pinecone; vector mode embeds locally.")

query = st.text_input("Your question", placeholder="Ask me about apple(s) or BMS")
col_search, col_clear = st.columns([1, 1])
with col_search:
    go = st.button("Search")
with col_clear:
    if st.button("Clear"):
        st.rerun()

if go and index_name and query.strip():
    idx = open_index(index_name)

    with st.spinner("Searching…"):
        try:
            mode_used = ""
            results = None

            if query_mode in ["Inference (server-side)", "Auto"]:
                try:
                    inference_model_to_use = SPECIAL_INFERENCE_MODEL if force_inference_for_special else None
                    results = try_inference_search(
                        idx, namespace, query, top_k, model_name=inference_model_to_use)
                    mode_used = "server-side (inference)"
                except Exception as e:
                    if query_mode == "Inference (server-side)":
                        raise
                    results = vector_query(
                        idx, namespace, query, top_k, EMBED_MODEL or DEFAULT_EMBED_MODEL)
                    mode_used = "client-side (vector)"
            else:
                results = vector_query(
                    idx, namespace, query, top_k, EMBED_MODEL or DEFAULT_EMBED_MODEL)
                mode_used = "client-side (vector)"
            matches = normalize_matches(results)
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

    st.caption(f"Mode used: **{mode_used}**")

    if not matches:
        st.info("No results.")
        st.stop()

    # Special index: show results only (no LLM answer unless user opts in)
    if force_inference_for_special and not generate_llm_answer_for_special:
        st.header("Results")
        for i, m in enumerate(matches, start=1):
            with st.container():
                st.markdown(f"**{i}. Score:** {m.get('score')}")
                snippet = m.get("text") or "_(no text in metadata)_"
                st.write(snippet)
                src = m.get("source") or "—"
                st.caption(f"ID: {m.get('id') or '—'} • Source: {src}")
                st.markdown("---")
    else:
        # Generate answer for other indexes, or if explicitly enabled for the special index
        snippets = [m.get("text", "") for m in matches if m.get("text")]
        context = build_context(snippets)
        try:
            answer = answer_question(query, context)
        except Exception as e:
            answer = f"(Answer generation failed) {e}"

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Results")
        for i, m in enumerate(matches, start=1):
            with st.container():
                st.markdown(f"**{i}. Score:** {m.get('score')}")
                snippet = m.get("text") or "_(no text in metadata)_"
                st.write(snippet)
                src = m.get("source") or "—"
                st.caption(f"ID: {m.get('id') or '—'} • Source: {src}")
                st.markdown("---")

    with st.expander("Raw response"):
        st.write(results)
else:
    st.info("Type a question and press **Search** to run a query.")
