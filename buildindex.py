# buildindex.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()  # loads variables from a .env file in the project root

api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise RuntimeError(
        "PINECONE_API_KEY is not set. Check your .env or environment.")

pc = Pinecone(api_key=api_key)

index_name = "docs-from-s3"
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
