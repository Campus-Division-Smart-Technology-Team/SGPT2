# 🦍 Alfred

A Streamlit-based chatbot application for the University of Bristol's Campus Innovation Technology team that provides intelligent search across Building Management Systems (BMS) documentation and Fire Risk Assessments (FRAs) with building-aware search capabilities.

Alfred ingests documents from **AWS S3**, converts them into **embeddings with OpenAI**, stores them in a **Pinecone** index, and uses a **Streamlit-based chat interface**. Alfred supports multiple document types (PDF, DOCX, TXT, JSON, CSV, Markdown), with intelligent building name detection, fuzzy matching, and comprehensive metadata management for retrieval-augmented generation (RAG).

## 🥇 Features

- **Building-Aware Search**: Dynamic building name cache with fuzzy matching (80% threshold) across multiple metadata fields
- **Smart Query Classification**: Automatically handles greetings, about queries, and farewells without API calls
- **Two-Stage Search Strategy**: 
  - Stage 1: Metadata-filtered search for building-specific queries
  - Stage 2: Semantic search with building-based result boosting
- **Business Term Recognition**: Automatic detection and expansion of technical terms (BMS, FRA, AHU, etc.)
- **Enhanced Date Detection**: Intelligently finds publication dates from document sources with priority-based extraction
- **Multiple Document Types**: 
  - Fire Risk Assessments (FRA)
  - Building Management System (BMS) operational documents
  - Planon property/building data
- **Quality Control**: Minimum relevance threshold (0.3) to prevent low-quality responses
- **Professional UI**: University of Bristol branding with WCAG 2.2 AA accessibility compliance

## 📁 File Structure

```
├── main.py                  # Main Streamlit application with building cache initialization
├── config.py                # Configuration settings and environment variables
├── clients.py               # Pinecone and OpenAI client initialization
├── query_classifier.py      # Smart query classification with pre-compiled patterns
├── pinecone_utils.py        # Pinecone search utilities and helpers
├── date_utils.py            # Publication date parsing with pre-compiled regex patterns
├── answer_generation.py     # OpenAI answer generation with building-aware context
├── search_operations.py     # Federated search with building filtering and fuzzy matching
├── ui_components.py         # Streamlit UI components with building cache status display
├── building_utils.py        # Building name extraction with fuzzy matching and n-gram fallback
├── business_terms.py        # Business terminology mapping (BMS, FRA, AHU, etc.)
├── batch_ingest.py          # Multi-threaded batch ingestion with alias support
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## 📦 Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Create a `.env` file or set environment variables:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   OPENAI_API_KEY=your_openai_api_key
   AWS_ACCESS_KEY_ID=your_aws_access_key  # For ingestion scripts
   AWS_SECRET_ACCESS_KEY=your_aws_secret  # For ingestion scripts
   
   # Optional configuration
   ANSWER_MODEL=gpt-4o-mini               # Defaults to gpt-4o-mini
   DEFAULT_EMBED_MODEL=text-embedding-3-small  # Defaults to text-embedding-3-small
   LOG_LEVEL=INFO                         # Logging level
   ```

3. **Run the Application**:
   ```bash
   streamlit run main.py
   ```

## 🔧 Configuration

Key settings in `config.py`:

- `TARGET_INDEXES`: List of Pinecone indexes to search (default: `["operational-docs"]`)
- `MIN_SCORE_THRESHOLD`: Minimum relevance score (0.3)
- `SEARCH_ALL_NAMESPACES`: Whether to search all namespaces or just the default
- `INDEX_CONFIGS`: Per-index configuration for embedding model and dimension

### Building Name Recognition

Alfred uses a dynamic cache populated from Pinecone indexes at startup:

- **Canonical Names**: Official building names from property data
- **Aliases**: Common abbreviations and alternative names (e.g., "BDFI" → "65 Avon Street")
- **Fuzzy Matching**: 80% similarity threshold for flexible matching
- **Multiple Metadata Fields**: Checks `canonical_building_name`, `building_name`, `Property names`, `UsrFRACondensedPropertyName`, and more

## 🏭 Usage

### Query Types

The app intelligently handles different types of queries:

1. **Greetings**: "Hi", "Hello Alfred" → Direct friendly response
2. **About Queries**: "What can you do?" → Information about capabilities
3. **Building-Specific Queries**: Automatically detects building names or abbreviations
4. **Search Queries**: Domain-specific questions → Building-aware federated search with AI-generated answers
5. **Gratitude/Farewells**: "Thanks", "Bye" → Appropriate responses

### Search Features

- **Building Detection**: Automatically extracts building names from queries (e.g., "Senate House", "BDFI", "1-9 Old Park Hill")
- **Two-Stage Search Strategy**:
  - Stage 1: Metadata-filtered search when building is detected
  - Stage 2: Semantic search with building-based boosting (fallback)
- **Fuzzy Matching**: 80% similarity threshold for building name matching
- **Document Type Boosting**: Prioritizes relevant document types (BMS, FRA, Planon)
- **Date Intelligence**: Finds the most recent publication/update date with priority-based extraction
- **Quality Control**: Won't provide answers if relevance score is below threshold

### Example Queries

**Building-Specific Queries**:
- "How many floors does Senate House have?"
- "Tell me about the BMS in BDFI"  # BDFI = 65 Avon Street
- "List fire risks at 1-9 Old Park Hill"
- "What is the frost protection sequence in Berkeley Square?"

**Cross-Building Queries**:
- "Compare the AHU controllers in Senate House and Berkeley Square"
- "What buildings have Trend IQ4 systems?"

**General Queries**:
- "Hello Alfred"
- "What can you help with?"
- "How does frost protection work?"

**Business Term Queries** (automatically expanded):
- "FRA for Senate House" → "Fire Risk Assessment for Senate House"
- "AHU in BDFI" → "Air Handling Unit in 65 Avon Street"
- "BMS access levels" → "Building Management System access levels"

## 💻 Technical Details

### Search Strategy

1. **Query Classification**: Determines if search is needed
2. **Building Extraction**: Detects building names using patterns and n-gram fallback
3. **Business Term Detection**: Identifies and expands technical terms
4. **Two-Stage Search**:
   - **Stage 1**: Metadata-filtered search with building filter
   - **Stage 2**: Semantic search with building boosting (3x for Stage 2, 1.5x for Stage 1)
5. **Result Deduplication**: Removes duplicate results keeping highest scores
6. **Date Enhancement**: Searches for publication dates across source documents
7. **Answer Generation**: Uses OpenAI to synthesize building-aware responses

### Building Name Detection

**Extraction Methods**:
1. **Pattern Matching**: Pre-compiled regex patterns for common phrases
2. **N-gram Fallback**: Tries 2-4 word combinations when patterns fail (handles lowercase queries)
3. **Validation**: Fuzzy matching against known building names from cache

**Matching Strategies** (in priority order):
1. Exact match in aliases cache
2. Exact match in canonical names
3. Substring matching (bidirectional)
4. Fuzzy matching against all metadata field variations (80% threshold)
5. Difflib fuzzy matching (80% threshold)

### Date Detection

The system uses multiple strategies with priority scoring:

**Priority Levels** (15 = highest, 3 = lowest):
- **15**: Explicitly labeled dates ("Last Updated", "Date Revised")
- **14**: Document header patterns
- **13**: Version dates
- **12**: General labeled dates
- **10**: Standard text dates with ordinals
- **6**: Numeric and ISO date formats
- **3**: Copyright/version years (low priority)

**Validation**:
- Date must not be in the future
- Date must be after 1990
- Date must be within last 30 years
- Checks multiple metadata fields: `review_date`, `updated`, `revised`, `date`, `document_date`

### Metadata Fields

**Building Names** (checked in priority order):
- `canonical_building_name`
- `building_name`
- `Property names`
- `UsrFRACondensedPropertyName`
- `building_aliases`

**Document Types**:
- `fire_risk_assessment`: Fire Risk Assessment documents
- `operational_doc`: BMS operational and O&M documents
- `planon_data`: Property and building condition data
- `unknown`: Unclassified documents

### Business Terms

Pre-defined terms with automatic expansion:

| Term | Full Name | Document Type |
|------|-----------|---------------|
| FRA | Fire Risk Assessment | fire_risk_assessment |
| BMS | Building Management System | operational_doc |
| AHU | Air Handling Unit | operational_doc |
| HVAC | Heating, Ventilation, and Air Conditioning | operational_doc |
| IQ4 | IQ4 Controller | operational_doc |
| O&M | Operations & Maintenance | operational_doc |
| DesOps | Description of Operations | operational_doc |
| Planon | Planon Property Management | planon_data |

## 📊 Data Ingestion

### Batch Ingest Script (`batch_ingest.py`)

Production-ready multi-threaded ingestion:

```bash
python batch_ingest.py --bucket your-bucket --prefix path/to/docs --workers 4
```

**Features**:
- Parallel processing with configurable workers
- Skip existing vectors option
- Building metadata cache
- Comprehensive statistics tracking
- Enhanced error handling and retry logic

**Environment Variables**:
```
BUCKET=your-bucket-name
PREFIX=path/to/docs
INDEX_NAME=operational-docs
NAMESPACE=                    # Optional, defaults to None
EMBED_MODEL=text-embedding-3-small
DIMENSION=1536
CHUNK_TOKENS=500
CHUNK_OVERLAP=50
EMBED_BATCH=64
UPSERT_BATCH=64
MAX_FILE_MB=0                 # 0 = no limit
SKIP_EXISTING=true            # Skip already indexed documents
MAX_WORKERS=1                 # Number of parallel workers
```

## 🚀 Deployment

For Streamlit Cloud deployment:

1. Push code to GitHub repository
2. Connect repository to Streamlit Cloud
3. Add secrets in Streamlit Cloud dashboard:
   - `PINECONE_API_KEY`
   - `OPENAI_API_KEY`
   - Any other environment variables from `config.py`

**Important**: Building cache is initialised automatically at startup from Pinecone indexes.

## ♿ Accessibility

The application follows WCAG 2.2 AA guidelines with:
- Semantic HTML structure
- Proper color contrast
- Screen reader compatibility
- Keyboard navigation support
- Accessible form controls
- Footer with accessibility statement

## 🐛 Troubleshooting

### Building Cache Not Initialising

**Symptoms**: Warning "Building cache could not be initialised"

**Solutions**:
1. Ensure `planon_data` documents exist in at least one target index
2. Check that documents have `canonical_building_name` or `building_name` metadata
3. Verify Pinecone connection and credentials
4. Check logs for specific error messages

### Building Name Not Detected

**Symptoms**: Queries don't filter by building correctly

**Solutions**:
1. Check if building name is in cache (see sidebar cache status)
2. Try using exact building name or known abbreviation
3. Use quotes or explicit phrases: "at Senate House", "in BDFI"
4. Check building aliases in Planon data

### Low Search Results

**Symptoms**: "I couldn't find any relevant information"

**Solutions**:
1. Rephrase query to be more specific
2. Use building names or abbreviations
3. Include relevant business terms (BMS, FRA, AHU)
4. Check that documents are properly indexed
5. Verify minimum score threshold (0.3) is appropriate

### Date Information Missing

**Symptoms**: "Publication date unknown"

**Solutions**:
1. Ensure source documents contain dates in recognised formats
2. Check that documents are properly chunked (dates in early chunks)
3. Verify document type is correctly detected
4. Review date validation rules (1990-present, max 30 years old)

## 📝 License

Internal use only - University of Bristol Smart Technology Team

## 👥 Contact

For issues or questions, contact the **Campus Innovation Technology Data Team** at the University of Bristol.