#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Q&A over a Pinecone index (cleaned version).

Behavior:
- When the selected index is exactly "llama-text-embed-v2-index", use Pinecone
  server-side inference with the attached model (llama-text-embed-v2) to search
  and ONLY display the search results (no LLM-generated answer) unless the user
  explicitly opts in.
- Other indexes behave as usual: you can use server-side inference when available
  or client-side vector search; the retrieved context is then used to generate an
  answer with an LLM (configurable via env).

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

from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import json

import streamlit as st
from pinecone import Pinecone
from openai import OpenAI

# ---------- App & env ----------
st.set_page_config(page_title="Apples & BMS", page_icon="🔎", layout="wide")

# Try to load a local .env if python-dotenv is available; otherwise ignore
try:  # noqa: SIM105
    from dotenv import load_dotenv  # type: ignore
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

DEFAULT_EMBED_MODEL = os.getenv("DEFAULT_EMBED_MODEL", "text-embedding-3-small")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4o-mini")

# ---------- Clients (cached) ----------


@st.cache_resource
def get_oai() -> OpenAI:
    return OpenAI()


@st.cache_resource
def get_pc() -> Pinecone:
    return Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))


oai = get_oai()
pc = get_pc()


# ---------- Index helpers (cached) ----------


@st.cache_data(ttl=30)
def cached_list_index_names(pc_client: Pinecone) -> List[str]:
    try:
        idxs = pc_client.list_indexes()
    except Exception:
        return []
    if hasattr(idxs, "names"):
        return list(idxs.names())
    if isinstance(idxs, dict) and "indexes" in idxs:
        return [i.get("name") for i in idxs.get("indexes", []) if i.get("name")]
    return list(idxs) if isinstance(idxs, (list, tuple)) else []


def list_index_names() -> List[str]:
    return cached_list_index_names(pc)


@st.cache_data(ttl=30)
def list_namespaces_for_index(idx) -> List[str]:
    """Return namespaces present in the index via describe_index_stats()."""
    try:
        stats = idx.describe_index_stats()
        ns = (stats.get("namespaces") if isinstance(stats, dict) else None) or {}
        names = sorted(list(ns.keys()))
        return names or [""]
    except Exception:
        # If stats not available, just expose default namespace
        return [""]


def open_index(name: str):
    return pc.Index(name)


# ---------- Search utilities ----------


def embed_texts(texts: List[str], model: str) -> List[List[float]]:
    res = oai.embeddings.create(model=model, input=texts)
    return [d.embedding for d in res.data]


def vector_query(idx, namespace: str, query: str, k: int, embed_model: str, flt: Optional[dict] = None) -> Dict[str, Any]:
    """Client-side embedding + vector query."""
    vector = embed_texts([query], embed_model)[0]
    return idx.query(vector=vector, top_k=k, namespace=namespace or "", include_metadata=True, filter=flt)


def try_inference_search(idx, q: str, k: int, model_name: Optional[str] = None, ns: str = "", flt: Optional[dict] = None) -> Dict[str, Any]:
    """Attempt Pinecone inference search using modern SDKs first; fallback to manual embed+query on legacy."""
    # Newer SDK path with Index.search
    try:
        # 1) Preferred: direct Index.search
        try:
            return idx.search(inputs={"text": q}, top_k=k, include_metadata=True, namespace=ns or "", filter=flt)
        except TypeError:
            # Some SDKs take a single query dict
            return idx.search(namespace=ns or "", query={"inputs": {"text": q}, "top_k": k, "include_metadata": True, "filter": flt})
    except Exception:
        # fall through to legacy path
        pass

    # Compatibility path for older SDKs without Index.search
    try:
        embed_api = getattr(pc, "inference", None)
        if not embed_api:
            raise RuntimeError("Pinecone inference client not available in this SDK")
        embs = embed_api.embed(
            model=(model_name or SPECIAL_INFERENCE_MODEL),
            inputs=[q],
            parameters={"input_type": "query", "truncate": "END"},
        )
        first = embs.data[0]
        vec = first.get("values") if isinstance(first, dict) else getattr(first, "values", None)
        if vec is None:
            vec = first.get("embedding") if isinstance(first, dict) else getattr(first, "embedding", None)
        if vec is None:
            raise RuntimeError("Unexpected embeddings response shape; no vector values found")
    except Exception as e:
        raise RuntimeError(f"Server-side embedding failed: {e}")

    return idx.query(vector=vec, top_k=k, namespace=ns or "", include_metadata=True, filter=flt)


def _as_dict(obj: Any) -> Dict[str, Any]:
    return obj if isinstance(obj, dict) else getattr(obj, "to_dict", lambda: obj)()


def normalize_matches(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize Pinecone result shapes to a list of {id, score, metadata, text, source}."""
    if not data:
        return []

    # Common vector-query shape: {"matches": [...]}
    matches = data.get("matches") if isinstance(data, dict) else []
    if isinstance(matches, list) and matches:
        out: List[Dict[str, Any]] = []
        for m in matches:
            m = _as_dict(m)
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
            })
        return out

    return []


# ---------- Answering ----------


def build_context(snippets: List[str], max_chars: int = 6000) -> str:
    """Concatenate snippets with a hard character cap (not token-aware)."""
    buf: List[str] = []
    total = 0
    for s in snippets:
        if not s:
            continue
        if total + len(s) > max_chars:
            s = s[: max_chars - total]
        buf.append(s)
        total += len(s)
        if total >= max_chars:
            break
    return "\n\n---\n\n".join(buf)


def answer_question(question: str, context: str, model: Optional[str] = None, max_tokens: int = 400) -> str:
    model = model or ANSWER_MODEL
    sys_prompt = (
        "You are a concise, factual assistant. Use ONLY the provided context to answer. "
        "If the context lacks the answer, say you don't know and suggest what to search next."
    )
    msgs = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
    ]
    try:
        res = oai.chat.completions.create(model=model, messages=msgs, temperature=0.2, max_tokens=max_tokens)
        return (res.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[Answer generation failed: {e}]"


# ---------- UI ----------

st.title("🔎 Apples & BMS — Pinecone Search Console")

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
            default_ix = display_labels.index("(default)") if default_ns == "" else display_labels.index(default_ns)
        except ValueError:
            default_ix = 0
        selected_label = st.selectbox("Namespace", options=display_labels, index=default_ix, help="From index stats")
        namespace = "" if selected_label == "(default)" else selected_label
    else:
        # Fallback in case there are no indexes yet
        namespace = st.text_input("Namespace", value="")

    top_k = st.slider("Top K", min_value=1, max_value=25, value=5)

    force_inference_for_special = (index_name == SPECIAL_INFERENCE_INDEX)

    if force_inference_for_special:
        st.info("This index forces server-side inference using its attached model. By default, no LLM answer is generated.")
        generate_llm_answer_for_special = st.checkbox("Also generate an OpenAI answer (uses OPENAI_API_KEY)", value=False)
        query_mode = "Inference (server-side)"
        EMBED_MODEL = None
    else:
        query_mode = st.selectbox(
            "Query mode",
            options=["Auto", "Inference (server-side)", "Vector (client-side)"],
            index=0,
            help="Auto tries server-side inference first, then falls back to vector query.",
        )
        EMBED_MODEL = st.text_input(
            "Embedding model (Vector mode)",
            value=DEFAULT_EMBED_MODEL,
            help="Used only when 'Vector (client-side)' is chosen or Auto falls back.",
        )

    with st.expander("Optional filter (JSON)"):
        filter_str = st.text_area("Filter", value="", placeholder='e.g. {"author":"alice"}')
        filter_obj: Optional[dict] = None
        if filter_str.strip():
            try:
                filter_obj = json.loads(filter_str)
                st.caption("Filter parsed ✓")
            except Exception as e:
                st.error(f"Invalid JSON filter: {e}")
                filter_obj = None

    st.markdown("---")
    with st.expander("Diagnostics"):
        has_pc = bool(os.environ.get("PINECONE_API_KEY"))
        has_oai = bool(os.environ.get("OPENAI_API_KEY"))
        st.write(f"PINECONE_API_KEY: {'✓' if has_pc else '✗'}")
        st.write(f"OPENAI_API_KEY: {'✓' if has_oai else '✗'}")
        if query_mode in ("Auto", "Vector (client-side)") and not has_oai:
            st.warning("Vector mode and answer generation require OPENAI_API_KEY.")

# --- Query box ---
query = st.text_input("Your question", placeholder="Ask me about apple(s) or BMS")
col_search, col_clear = st.columns([1, 1])
with col_search:
    go = st.button("Search")
with col_clear:
    if st.button("Clear"):
        st.rerun()

if go and not query.strip():
    st.warning("Please enter a question.")
    go = False

if go:
    # Preflight checks
    missing = []
    if not os.environ.get("PINECONE_API_KEY"):
        missing.append("PINECONE_API_KEY")
    need_oai = (query_mode == "Vector (client-side)") or (not force_inference_for_special) or (force_inference_for_special and 'generate_llm_answer_for_special' in locals() and generate_llm_answer_for_special)
    if need_oai and not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if missing:
        st.warning("Missing required environment variables: " + ", ".join(missing))

    # Open index now (may raise)
    try:
        idx = open_index(index_name)
    except Exception as e:
        st.error(f"Could not open index '{index_name}': {e}")
        st.stop()

    # Run search according to mode
    results: Dict[str, Any] = {}
    matches: List[Dict[str, Any]] = []
    mode_used = ""

    if force_inference_for_special:
        try:
            results = try_inference_search(idx, query, top_k, model_name=SPECIAL_INFERENCE_MODEL, ns=namespace, flt=filter_obj)
            mode_used = "server-side inference (forced)"
            matches = normalize_matches(results)
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()
    else:
        try:
            if query_mode == "Auto":
                try:
                    results = try_inference_search(idx, query, top_k, ns=namespace, flt=filter_obj)
                    mode_used = "server-side inference"
                except Exception:
                    results = vector_query(idx, namespace, query, top_k, EMBED_MODEL or DEFAULT_EMBED_MODEL, flt=filter_obj)
                    mode_used = "client-side (vector)"
            elif query_mode == "Inference (server-side)":
                results = try_inference_search(idx, query, top_k, ns=namespace, flt=filter_obj)
                mode_used = "server-side inference"
            else:
                results = vector_query(idx, namespace, query, top_k, EMBED_MODEL or DEFAULT_EMBED_MODEL, flt=filter_obj)
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
    if force_inference_for_special and not ('generate_llm_answer_for_special' in locals() and generate_llm_answer_for_special):
        st.header("Results")
        for i, m in enumerate(matches, start=1):
            with st.container():
                score = m.get("score")
                score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "—"
                st.markdown(f"**{i}. Score:** {score_str}")
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
            answer = f"[Answer generation failed: {e}]"

        st.header("Answer")
        st.write(answer)

        st.subheader("Results")
        for i, m in enumerate(matches, start=1):
            with st.container():
                score = m.get("score")
                score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "—"
                st.markdown(f"**{i}. Score:** {score_str}")
                snippet = m.get("text") or "_(no text in metadata)_"
                st.write(snippet)
                src = m.get("source") or "—"
                st.caption(f"ID: {m.get('id') or '—'} • Source: {src}")
                st.markdown("---")

    with st.expander("Index details"):
        st.write(f"**Index:** {index_name or '—'}")
        if force_inference_for_special:
            st.write(f"**Forced inference model:** {SPECIAL_INFERENCE_MODEL}")
        st.caption("Server-side inference runs inside Pinecone; vector mode embeds locally.")

    with st.expander("Raw response"):
        try:
            st.json(results)
        except Exception:
            st.write(results)
else:
    st.info("Type a question and press **Search** to run a query.")
