#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pinecone utilities for index management and search operations.
"""

from typing import Any, Dict, List, Optional
import logging
from clients import pc, oai
from config import DEFAULT_NAMESPACE, DEFAULT_EMBED_MODEL, SPECIAL_INFERENCE_MODEL


def list_index_names() -> List[str]:
    """Get list of available Pinecone index names."""
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
    """Open and return a Pinecone index."""
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


def embed_texts(texts: List[str], model: str) -> List[List[float]]:
    """Generate embeddings for a list of texts using OpenAI."""
    res = oai.embeddings.create(model=model, input=texts)
    return [d.embedding for d in res.data]


def vector_query(idx, namespace: str, query: str, k: int, embed_model: str) -> Dict[str, Any]:
    """Perform vector search using client-side embeddings."""
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
    """Convert object to dictionary if possible."""
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
                # Skip publication_date from metadata as it's misleading
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
                # Skip publication_date from metadata
            })
        return out

    return []
