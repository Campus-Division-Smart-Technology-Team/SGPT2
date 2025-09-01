# 📖 S3 → Pinecone Document Search & Q&A

This project ingests documents from **AWS S3**, converts them into **embeddings with OpenAI**, stores them in **Pinecone**, and provides a **Streamlit-based Q&A interface**. It supports multiple document types (PDF, DOCX, TXT, JSON, CSV, Markdown), chunking, batching, and metadata storage for retrieval-augmented generation (RAG).

---

## :file_folder: Project Structure
  ```
  .
├── ingest.py # Ingests files from S3 → OpenAI embeddings → Pinecone
├── buildindex.py # Ensures Pinecone index exists (creates if missing)
├── query.py # Streamlit app for semantic search & Q&A
├── main.py # Simple OpenAI example (story generation)
├── test.py # Test env & keys (AWS, Pinecone, OpenAI)
├── Requirements.txt # Python dependencies
└── .env # Store your API keys & config
  ```
---

## :gear: Requirements

See `Requirements.txt`.  
Key dependencies include:

- [pinecone-client](https://docs.pinecone.io/) (`pinecone-client`)
- [openai](https://pypi.org/project/openai/) (`openai`)
- [boto3](https://boto3.amazonaws.com/) (for AWS S3 access)
- [pypdf](https://pypi.org/project/pypdf/) & [python-docx](https://pypi.org/project/python-docx/) (document parsing)
- [tiktoken](https://github.com/openai/tiktoken) (tokenization & chunking)
- [streamlit](https://streamlit.io/) (web interface)
- [python-dotenv](https://pypi.org/project/python-dotenv/) (env config)

---

## :rocket: Setup

1. Clone the repo:
   ```
   bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
   ```
3. Install dependencies:
   ```
   pip install -r Requirements.txt
   ```

4. Configure your `.env` file:
   ```
   # Required
   OPENAI_API_KEY=your-openai-api-key
   PINECONE_API_KEY=your-pinecone-api-key
   
   # AWS (for ingest.py)
   AWS_ACCESS_KEY_ID=your-aws-key
   AWS_SECRET_ACCESS_KEY=your-aws-secret
   AWS_DEFAULT_REGION=us-east-1
   
   # Optional overrides
   BUCKET=your-bucket-name
   PREFIX=docs/
   INDEX_NAME=docs-from-s3
   NAMESPACE=default
   ```

## :factory: Usage
1. Build (or verify) Pinecone index
```
python buildindex.py
```
   
2. Ingest documents from S3
```
python ingest.py --bucket your-bucket-name --prefix your-folder/
```
  > Extracts text from supported files (`.pdf`, `.docx`, `.txt`, `.md`, `.csv`, `.json`)
  > Splits into overlapping chunks (default 500 tokens)
  > Embeds via OpenAI & upserts to Pinecone

3. Run Q&A interface
```
streamlit run query.py
```
  >Enter your question in the UI
  >Results are retrieved from Pinecone
  >OpenAI synthesises a final answer citing sources

## :wrench: Customisation
1. Change defaults in `ingest.py` (e.g., `CHUNK_TOKENS`, `EMBED_MODEL`, `UPSERT_BATCH`).
2. Adjust Pinecone `INDEX_NAME` and `NAMESPACE` in .env.
3. Extend `EXT_WHITELIST` in `ingest.py` to support more file formats.

### :page_with_curl: Notes
1. The system stores actual text chunks in Pinecone metadata, enabling snippet-based answers.
2. Supports both server-side inference search (if index has attached model) and client-side vector search.
3. Backoff & retry logic is included for OpenAI and Pinecone API calls.
4. For large datasets, ingestion is batched, and upserts are chunked for efficiency.

