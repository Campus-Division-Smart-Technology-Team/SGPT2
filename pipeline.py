import os
import glob
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# Configuration
MODEL_PATH = "./models/arch-llama-3b-q4_0.bin"  # Path to your quantized model
CHUNK_SIZE = 500  # Number of tokens per text chunk
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence-Transformers model
INDEX_FILE = "docs_index.faiss"  # FAISS index file
DOCS_PATH = "./docs"  # Directory containing text documents (txt, md, etc.)

# 1. Initialize models
embedder = SentenceTransformer(EMBEDDING_MODEL)
llm = Llama(model_path=MODEL_PATH, n_ctx=2048)

def read_documents(path: str) -> List[str]:
    """
    Reads all text files from a directory and returns a list of raw text strings.
    """
    docs = []
    for ext in ("*.txt", "*.md", "*.rst"):
        for filepath in glob.glob(os.path.join(path, ext)):
            with open(filepath, 'r', encoding='utf-8') as f:
                docs.append(f.read())
    return docs

# 2. Chunking
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def chunk_text(text: str, max_tokens: int = CHUNK_SIZE) -> List[str]:
    """
    Splits a document into chunks of approximately `max_tokens` tokens.
    """
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokenizer.decode(tokens[i:i + max_tokens])
        chunks.append(chunk)
    return chunks

# 3. Ingest and prepare chunks
all_chunks: List[str] = []
docs = read_documents(DOCS_PATH)
for doc in docs:
    all_chunks.extend(chunk_text(doc))

# 4. Compute embeddings and build FAISS index
def build_faiss_index(chunks: List[str], index_path: str):
    # Compute embeddings in batches
    embeddings = embedder.encode(chunks, convert_to_numpy=True, batch_size=32, show_progress_bar=True)
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return index

if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    index = build_faiss_index(all_chunks, INDEX_FILE)

# 5. RAG query function
def rag_query(query: str, k: int = 5, max_gen_tokens: int = 256) -> str:
    # a) Embed the query
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    # b) Retrieve top-k
    D, I = index.search(q_emb, k)
    retrieved = [all_chunks[idx] for idx in I[0]]

    # c) Build prompt
    context = "\n\n---\n\n".join(retrieved)
    prompt = (
        "You are a helpful assistant. Use the following documents to answer the question.\n\n"
        + context
        + f"\n\nQuestion: {query}\nAnswer:"
    )

    # d) Generate answer
    response = llm(prompt=prompt, max_tokens=max_gen_tokens, temperature=0.7, top_p=0.9)
    return response["choices"][0]["text"].strip()

# 6. Interactive test
def main():
    print("RAG Pipeline Ready. Enter your query:")
    while True:
        query = input("Q: ")
        if query.lower() in ['exit', 'quit']:
            break
        answer = rag_query(query)
        print(f"A: {answer}\n")

if __name__ == "__main__":
    main()
