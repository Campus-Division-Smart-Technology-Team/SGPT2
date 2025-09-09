# :gorilla: Ask Alfred

A Streamlit-based chatbot application for the University of Bristol that provides intelligent search across multiple Pinecone indexes with enhanced query classification and publication date detection.

Alfred ingests documents from **AWS S3**, converts them into **embeddings with OpenAI**, stores them in **Pinecone**, and uses a **Streamlit-based chat interface**. Alfred supports multiple document types (PDF, DOCX, TXT, JSON, CSV, Markdown), chunking, batching, and metadata storage for retrieval-augmented generation (RAG).

## 	:1st_place_medal: Features

- **Smart Query Classification**: Automatically handles greetings, about queries, and farewells without API calls
- **Federated Search**: Searches across multiple Pinecone indexes ("apples" and "test")
- **Enhanced Date Detection**: Intelligently finds publication dates from document sources
- **Quality Control**: Minimum relevance threshold to prevent low-quality responses
- **University of Bristol Branding**: Professional UI with accessibility features

## :file_folder: File Structure

```
├── main.py                # Main Streamlit application
├── config.py              # Configuration settings and environment variables
├── clients.py             # Pinecone and OpenAI client initialisation
├── query_classifier.py    # Smart query classification logic
├── pinecone_utils.py      # Pinecone search utilities and helpers
├── date_utils.py          # Publication date parsing and search
├── answer_generation.py   # OpenAI answer generation with source dates
├── search_operations.py   # Federated search across multiple indexes
├── ui_components.py       # Streamlit UI components and styling
├── requirements.txt       # Python dependencies
├── others                 # directory containing legacy files
└── README.md              # This file
```

## :package: Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Create a `.env` file or set environment variables:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   OPENAI_API_KEY=your_openai_api_key
   ANSWER_MODEL=gpt-4o-mini  # Optional, defaults to gpt-4o-mini
   DEFAULT_EMBED_MODEL=text-embedding-3-small  # Optional
   ```

3. **Run the Application**:
   ```bash
   streamlit run main.py
   ```

## :wrench: Configuration

Key settings in `config.py`:

- `TARGET_INDEXES`: List of Pinecone indexes to search
- `MIN_SCORE_THRESHOLD`: Minimum relevance score (0.3)
- `SPECIAL_INFERENCE_INDEX`: Index that uses server-side inference
- `SEARCH_ALL_NAMESPACES`: Whether to search all namespaces or just the default

## :factory: Usage

### Query Types

The app intelligently handles different types of queries:

1. **Greetings**: "Hi", "Hello Alfred" → Direct friendly response
2. **About Queries**: "What can you do?" → Information about capabilities
3. **Search Queries**: Domain-specific questions → Federated search with AI-generated answers
4. **Gratitude/Farewells**: "Thanks", "Bye" → Appropriate responses

### Search Features

- **Federated Search**: Searches across multiple indexes simultaneously
- **Smart Routing**: "apples" index uses server-side inference, others try inference, then fallback to vector search
- **Date Intelligence**: Finds the most recent publication/update date from source documents
- **Quality Control**: Won't provide answers if relevance score is below threshold

### Example Queries

**Apple Topics**:
- "What is Apple's flagship product?"
- "Tell me about different types of apples"

**BMS Topics**:
- "How does the frost protection sequence operate?"
- "What access levels are defined for controllers?"

## :computer: Technical Details

### Search Strategy

1. **Query Classification**: Determines if search is needed
2. **Federated Search**: Queries multiple indexes in parallel
3. **Result Merging**: Combines and ranks results by relevance score
4. **Date Enhancement**: Searches for publication dates across source documents
5. **Answer Generation**: Uses OpenAI to synthesise responses with date context

### Date Detection

The system uses multiple strategies to find publication dates:
- Searches all chunks from the same document source
- Supports various date formats (dots, slashes, text)
- Validates dates for reasonableness
- Prioritises context-aware patterns

### Quality Control

- Minimum score threshold prevents irrelevant responses
- Fallback to "I don't know" for low-quality matches
- Enhanced error handling and logging

## :rocket: Deployment

For Streamlit Cloud deployment:

1. Push code to the GitHub repository
2. Connect repository to Streamlit Cloud
3. Add secrets in the Streamlit Cloud dashboard:
   - `PINECONE_API_KEY`
   - `OPENAI_API_KEY`

## :accessibility: Accessibility

The application follows WCAG 2.2 AA guidelines with:
- Semantic HTML structure
- Proper colour contrast
- Screen reader compatibility
- Keyboard navigation support
