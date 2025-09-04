import os
import io
import json
import re
import hashlib
import time
import math
import random
import argparse
import logging
from typing import Iterable, List, Dict, Tuple
from datetime import datetime

import boto3
from botocore.exceptions import ClientError
from botocore.config import Config as BotoConfig
from pypdf import PdfReader
from docx import Document as DocxDocument
import tiktoken
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from openai import APIError, RateLimitError

# ---------------- Env & constants ----------------
load_dotenv()

# ---- logging ----
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

# -------------- Config (edit or override via env/CLI) --------------
DEFAULT_BUCKET = os.getenv("BUCKET", "desopsus")
DEFAULT_PREFIX = os.getenv("PREFIX", "")             # e.g., "docs/"
INDEX_NAME = os.getenv("INDEX_NAME", "bms")
NAMESPACE = os.getenv("NAMESPACE") or None
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # 1536 dims
DIMENSION = int(os.getenv("DIMENSION", "1536"))
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
# smaller to reduce backoffs
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))
UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", "64"))
MAX_FILE_MB = float(os.getenv("MAX_FILE_MB", "0"))   # 0 disables size guard
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "30"))  # seconds

EXT_WHITELIST = {"txt", "md", "csv", "json", "pdf", "docx"}

# ---------------- Clients ----------------
s3 = boto3.client(
    "s3",
    config=BotoConfig(
        connect_timeout=5,
        read_timeout=60,
        retries={"max_attempts": 5}
    ),
)
pc = Pinecone(api_key=PINECONE_API_KEY)
oai = OpenAI(api_key=OPENAI_API_KEY)
enc = tiktoken.get_encoding("cl100k_base")

# Ensure index exists
existing = {i["name"] for i in pc.list_indexes()}
if INDEX_NAME not in existing:
    logging.info(f"Creating Pinecone index '{INDEX_NAME}' (dim={DIMENSION})…")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(INDEX_NAME)

# ---------------- Helpers ----------------


def backoff_sleep(attempt: int, base: float = 0.5, cap: float = 10.0):
    t = min(cap, base * (2 ** attempt)) * (0.5 + random.random())
    time.sleep(t)


def list_s3_objects(bucket: str, prefix: str = "") -> List[Dict]:
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    out = []
    for p in pages:
        for obj in p.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            if ext(key) not in EXT_WHITELIST:
                continue
            out.append(obj)  # keep LastModified, Size, ETag
    return out


def fetch_bytes(bucket: str, key: str) -> bytes:
    r = s3.get_object(Bucket=bucket, Key=key)
    return r["Body"].read()


def ext(key: str) -> str:
    return key.lower().rsplit(".", 1)[-1] if "." in key else ""


def extract_text(key: str, data: bytes) -> str:
    e = ext(key)
    if e in {"txt", "md", "csv", "json"}:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1", errors="ignore")
        if e == "json":
            try:
                text = json.dumps(json.loads(
                    text), ensure_ascii=False, indent=2)
            except Exception:
                pass
        return text

    if e == "pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n".join([p.extract_text() or "" for p in reader.pages])
        except Exception as ex:
            logging.warning(f"PDF extract failed for {key}: {ex}")
            return ""

    if e == "docx":
        try:
            doc = DocxDocument(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as ex:
            logging.warning(f"DOCX extract failed for {key}: {ex}")
            return ""

    return ""  # unsupported


def chunk_text(text: str, chunk_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP) -> List[str]:
    toks = enc.encode(text)
    chunks = []
    i = 0
    while i < len(toks):
        j = min(i + chunk_tokens, len(toks))
        chunk = enc.decode(toks[i:j])
        chunk = re.sub(r"\s+\n", "\n", chunk).strip()
        if chunk:
            chunks.append(chunk)
        i = max(0, j - overlap)
        if j == len(toks):
            break
    return chunks


def embed_texts_batch(texts: List[str], max_retries: int = 5) -> List[List[float]]:
    if not texts:
        return []
    for attempt in range(max_retries):
        try:
            resp = oai.embeddings.create(
                model=EMBED_MODEL,
                input=texts,
                timeout=OPENAI_TIMEOUT,
            )
            return [d.embedding for d in resp.data]
        except (RateLimitError, APIError) as e:
            logging.warning(
                f"Embedding API backoff (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise
            backoff_sleep(attempt)
        except Exception as e:
            logging.error(f"Embedding failed with unexpected error: {e}")
            raise


def make_id(bucket: str, key: str, chunk_idx: int) -> str:
    base = f"{bucket}:{key}:{chunk_idx}"
    return hashlib.md5(base.encode()).hexdigest()


def upsert_vectors(vectors: List[Tuple[str, List[float], Dict]]):
    for i in range(0, len(vectors), UPSERT_BATCH):
        batch = vectors[i:i+UPSERT_BATCH]
        for attempt in range(5):
            try:
                index.upsert(vectors=batch, namespace=NAMESPACE)
                break
            except Exception as e:
                logging.warning(f"Pinecone upsert retry {attempt+1}/5: {e}")
                if attempt == 4:
                    raise
                backoff_sleep(attempt)

# ---------------- Main ingest ----------------


def ingest_bucket(bucket: str, prefix: str = ""):
    t_start = time.time()
    objs = list_s3_objects(bucket, prefix)
    logging.info(f"Found {len(objs)} files under s3://{bucket}/{prefix}")

    pending: List[Tuple[str, List[float], Dict]] = []

    for o in objs:
        key = o["Key"]
        size = int(o.get("Size") or 0)
        last_modified = o.get("LastModified")
        size_mb = size / (1024 * 1024) if size else 0.0

        # optional size guard
        if MAX_FILE_MB > 0 and size_mb > MAX_FILE_MB:
            logging.warning(
                f"Skipping large file ({size_mb:.1f} MB > {MAX_FILE_MB} MB): {key}")
            continue

        logging.info(
            f"[1/5] Fetching: s3://{bucket}/{key} (size={size_mb:.2f} MB)")
        t0 = time.time()
        try:
            data = fetch_bytes(bucket, key)
        except ClientError as e:
            logging.error(f"ERROR get_object {key}: {e}")
            continue
        logging.info(
            f"Fetched {len(data)} bytes in {time.time()-t0:.2f}s; extracting…")

        t1 = time.time()
        text = extract_text(key, data)
        t_extract = time.time() - t1
        if not text.strip():
            logging.warning(
                f"Skipping (no text): {key} [extract {t_extract:.2f}s]")
            continue
        logging.info(
            f"[2/5] Extracted ~{len(text)} chars in {t_extract:.2f}s; chunking…")

        t2 = time.time()
        chunks = chunk_text(text)
        logging.info(
            f"[3/5] {key}: {len(chunks)} chunks (chunk={CHUNK_TOKENS}, overlap={CHUNK_OVERLAP}) in {time.time()-t2:.2f}s")

        # embed in batches
        for i in range(0, len(chunks), EMBED_BATCH):
            batch_chunks = chunks[i:i+EMBED_BATCH]
            logging.info(
                f"[4/5] Embedding chunks {i}-{i+len(batch_chunks)-1} of {len(chunks)}…")
            t3 = time.time()
            embeddings = embed_texts_batch(batch_chunks)
            logging.info(
                f"Embedded {len(embeddings)} chunks in {time.time()-t3:.2f}s")

            for j, (emb, chunk_str) in enumerate(zip(embeddings, batch_chunks)):
                cid = make_id(bucket, key, i + j)
                metadata = {
                    "bucket": bucket,
                    "key": key,
                    "chunk": i + j,
                    "source": f"s3://{bucket}/{key}",
                    "size": size,
                    "last_modified": last_modified.isoformat() if isinstance(last_modified, datetime) else str(last_modified),
                    "ext": ext(key),
                    "text": chunk_str,   # <-- store the actual chunk text
                }
                pending.append((cid, emb, metadata))

            if len(pending) >= UPSERT_BATCH:
                logging.info(f"[5/5] Upserting {len(pending)} vectors…")
                t4 = time.time()
                upsert_vectors(pending)
                logging.info(f"Upserted in {time.time()-t4:.2f}s")
                pending.clear()

            # upsert periodically
            if len(pending) >= UPSERT_BATCH:
                logging.info(f"[5/5] Upserting {len(pending)} vectors…")
                t4 = time.time()
                logging.info(f"Upserted in {time.time()-t4:.2f}s")
                pending.clear()

        logging.info(f"Finished file: {key} in {time.time()-t0:.2f}s total")

    if pending:
        logging.info(f"Final upsert of remaining {len(pending)} vectors…")
        t4 = time.time()
        upsert_vectors(pending)
        logging.info(f"Final upsert done in {time.time()-t4:.2f}s")
        pending.clear()

    logging.info(f"Ingestion complete in {time.time()-t_start:.2f}s.")

# ---------------- CLI ----------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Ingest S3 docs into Pinecone via OpenAI embeddings")
    p.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3 bucket name")
    p.add_argument("--prefix", default=DEFAULT_PREFIX,
                   help="S3 key prefix (folder)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest_bucket(args.bucket, args.prefix)
