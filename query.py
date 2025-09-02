import os
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

st.set_page_config(page_title="Pinecone Q&A", page_icon="🔎", layout="wide")
load_dotenv()

# ---------------- helpers for secrets/env ----------------


def get_env_key(name: str, alt: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name) or (os.getenv(alt) if alt else None)
    if val:
        return val
    try:
        return st.secrets[name]
    except Exception:
        return None


# ---------------- sidebar: connections ----------------
st.sidebar.header("Connections")
pc_key = st.sidebar.text_input(
    "Pinecone API Key", type="password", value=get_env_key("PINECONE_API_KEY") or "")
oa_key = st.sidebar.text_input("OpenAI API Key (needed for vector mode & answering)",
                               type="password", value=get_env_key("OPENAI_API_KEY") or "")

if not pc_key:
    st.sidebar.error("Pinecone API key required.")
    st.stop()


@st.cache_resource
def get_pc(api_key: str) -> Pinecone:
    return Pinecone(api_key=api_key)


@st.cache_resource
def get_oai(api_key: Optional[str]) -> Optional[OpenAI]:
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


pc = get_pc(pc_key)
oai = get_oai(oa_key or None)

# ---------------- index selection & config ----------------


def list_index_names() -> List[str]:
    idxs = pc.list_indexes()
    if hasattr(idxs, "names"):
        return list(idxs.names())
    if isinstance(idxs, dict) and "indexes" in idxs:
        return [i["name"] for i in idxs["indexes"]]
    return list(idxs) if isinstance(idxs, (list, tuple)) else []


index_names = list_index_names()
index_name = st.sidebar.selectbox(
    "Index", options=index_names or ["docs-from-s3"], index=0)
namespace = st.sidebar.text_input("Namespace (blank for default)", value="")
top_k = st.sidebar.slider("Top K", min_value=1, max_value=25, value=5)

query_mode = st.sidebar.selectbox("Query mode", options=[
                                  "Auto", "Inference (server-side)", "Vector (client-side)"], index=0)

EMBED_MODEL = st.sidebar.text_input(
    "Embedding model (Vector mode)", value="text-embedding-3-small")
ANSWER_MODEL = st.sidebar.text_input("Answer model", value="gpt-4o-mini")

MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

# ---------------- utilities ----------------


def describe_index(name: str) -> Tuple[int, Dict[str, Any]]:
    d = pc.describe_index(name)
    dim = getattr(d, "dimension", None)
    raw = d if isinstance(d, dict) else getattr(
        d, "to_dict", lambda: {"_raw": str(d)})()
    if dim is None:
        dim = raw.get("dimension")
    return int(dim or 0), raw


def embed_query(text: str) -> List[float]:
    if not oai:
        raise RuntimeError("OpenAI key required for Vector mode.")
    resp = oai.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def try_inference_search(idx, ns: str, q: str, k: int):
    return idx.search(namespace=ns or "", inputs={"text": q}, top_k=k, include_metadata=True)


def vector_query(idx, ns: str, q: str, k: int):
    vec = embed_query(q)
    return idx.query(namespace=ns or "", vector=vec, top_k=k, include_metadata=True)


def normalize_matches(results: Any) -> List[Dict[str, Any]]:
    data = results.to_dict() if hasattr(results, "to_dict") else results
    if not isinstance(data, dict):
        return []
    return data.get("matches", []) or []


def synthesize_answer(question: str, snippets: List[Dict[str, str]]) -> str:
    if not oai:
        return "LLM not configured. Provide an OpenAI API key."
    blocks = []
    for i, s in enumerate(snippets, start=1):
        txt = (s.get("text") or "").strip()
        if not txt:
            continue
        src = s.get("source") or ""
        blocks.append(f"[{i}] Source: {src}\n{txt}")
    if not blocks:
        return "I couldn’t find relevant text in the top matches."
    prompt = (
        "Answer the user's question using ONLY the snippets. If not clearly stated, say you don't know.\n\n"
        + "\n\n---\n\n".join(blocks)
        + f"\n\n---\n\nQuestion: {question}\nAnswer concisely:"
    )
    resp = oai.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system",
                "content": "Be precise and cite snippet numbers like [1], [2]."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def trunc(s: str, n: int = 700) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n] + " …"


# ---------------- UI ----------------
st.title("Pinecone Q&A")

# one-time init for a deferred clear flag
if "_do_clear" not in st.session_state:
    st.session_state["_do_clear"] = False

# If last click requested a clear, do it BEFORE creating the input this run
if st.session_state["_do_clear"]:
    st.session_state.pop("user_query", None)
    st.session_state["_do_clear"] = False

# Text input
query = st.text_input(
    "Your question",
    placeholder="e.g., Tell me about BMS at the University of Bristol",
    key="user_query",
)

# Clear button appears AFTER the input, but defers the actual clearing to next run
if st.button("Clear", type="secondary", key="clear_btn"):
    st.session_state["_do_clear"] = True
    st.rerun()

if not query:
    st.info("Enter a question to search and answer.")
    st.stop()

# --- run search
idx = pc.Index(index_name)
index_dim, index_desc = describe_index(index_name)

with st.expander("Index details", expanded=False):
    st.write(f"**Dimension:** {index_dim}")
    st.json(index_desc)

if query_mode in ["Auto", "Vector (client-side)"]:
    chosen_dim = MODEL_DIMS.get(EMBED_MODEL)
    if chosen_dim and chosen_dim != index_dim:
        st.warning(
            f"Embedding dim ({chosen_dim}) doesn’t match index dim ({index_dim}). "
            "Use an embedding model with the same dimension as the index, or switch Query mode to "
            "'Inference (server-side)' if the index has an attached model."
        )

with st.spinner("Searching…"):
    try:
        mode_used = ""
        results = None
        if query_mode == "Inference (server-side)":
            results = try_inference_search(idx, namespace, query, top_k)
            mode_used = "server-side (inference)"
        elif query_mode == "Vector (client-side)":
            results = vector_query(idx, namespace, query, top_k)
            mode_used = "client-side (vector)"
        else:  # Auto
            try:
                results = try_inference_search(idx, namespace, query, top_k)
                mode_used = "server-side (inference)"
            except Exception:
                results = vector_query(idx, namespace, query, top_k)
                mode_used = "client-side (vector)"
        matches = normalize_matches(results)
    except Exception as e:
        st.error(f"Search failed: {e}")
        st.stop()

st.subheader(f"Results · {mode_used}")
st.caption(
    f"Top-K: {top_k} · Namespace: {namespace or '(default)'} · Matches: {len(matches)}")

if not matches:
    st.info(
        "No matches returned. If this index is your new 'docs-from-s3', make sure you've re-ingested after adding `metadata['text']`.")
    st.stop()

# Build snippets
snippets = []
for m in matches:
    md = m.get("metadata") or {}
    if not md:
        continue
    snippets.append({
        "text": md.get("text") or "",
        "source": md.get("source") or md.get("key") or "",
        "score": m.get("score")
    })

with st.expander("Metadata keys present in first match"):
    st.write(sorted(list((matches[0].get("metadata") or {}).keys())))

# Answer
st.subheader("Answer")
if not any(s.get("text") for s in snippets):
    st.info("No `text` in metadata for the top matches. Re-ingest with `text` added to metadata in ingest.py.")
else:
    st.write(synthesize_answer(query, snippets))

# Sources
st.subheader("Sources")
for i, s in enumerate(snippets, start=1):
    src = s.get("source") or ""
    txt = s.get("text") or ""
    if not txt and not src:
        continue
    st.markdown(f"**[{i}]** {src}")
    if txt:
        st.write(trunc(txt))
    st.markdown("---")

with st.expander("Raw matches"):
    st.json(matches)
