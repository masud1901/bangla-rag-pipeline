# Multilingual RAG System - AI Engineer (Level-1) Assessment

A Retrieval-Augmented Generation (RAG) system that supports both English and Bengali queries, built with FastAPI, Streamlit, and Docker. This project was developed as part of the AI Engineer (Level-1) Technical Assessment.

## 🎯 Assessment Requirements & Implementation Status

### ✅ **CORE TASK - COMPLETED**

**Requirement**: Build a basic RAG application that accepts user queries in English and Bangla
- ✅ **Implemented**: FastAPI backend with multilingual support
- ✅ **Implemented**: Streamlit GUI with Bengali sample questions
- ✅ **Implemented**: Dual embedding system (Cohere + OpenAI) for better multilingual understanding

**Requirement**: Retrieves relevant document chunks from a small knowledge base
- ✅ **Implemented**: ChromaDB vector database with 1,869 total vectors
- ✅ **Implemented**: Dual collections (Cohere: 903 vectors, OpenAI: 966 vectors)
- ✅ **Implemented**: Semantic similarity search with configurable top-k retrieval

**Requirement**: Generates answers based on the retrieved information
- ✅ **Implemented**: OpenAI GPT-3.5-turbo for answer generation
- ✅ **Implemented**: Cohere rerank for improved relevance
- ✅ **Implemented**: Context-aware prompt engineering

### ✅ **KNOWLEDGE BASE - COMPLETED**

**Requirement**: Use the following Bangla Book - HSC26 Bangla 1st paper
- ✅ **Implemented**: Successfully ingested `Bangla_book.pdf` (19MB)
- ✅ **Implemented**: Extracted 304 pages with OCR support
- ✅ **Implemented**: Created 966 chunks with semantic chunking

**Requirement**: Proper Pre-Processing & data cleaning for better chunk accuracy
- ✅ **Implemented**: Advanced OCR with Tesseract (Bengali + English)
- ✅ **Implemented**: OCR caching system for performance optimization
- ✅ **Implemented**: Text cleaning and normalization
- ✅ **Implemented**: Parallel processing for efficiency

**Requirement**: Document Chunking & Vectorize
- ✅ **Implemented**: RecursiveCharacterTextSplitter with Bengali-aware separators
- ✅ **Implemented**: Dual embedding generation (Cohere + OpenAI)
- ✅ **Implemented**: Semantic chunking strategy
- ✅ **Implemented**: Vector storage in ChromaDB

### ✅ **MEMORY MANAGEMENT - COMPLETED**

**Requirement**: Maintain Long-Short term memory
- ✅ **Short-Term Memory**: Streamlit session state for chat history
- ✅ **Short-Term Memory**: Conversation context preservation
- ✅ **Long-Term Memory**: ChromaDB with persistent storage
- ✅ **Long-Term Memory**: 1,869 total vectors from document corpus

### ❌ **SAMPLE TEST CASES - NEEDS VERIFICATION**

**Assessment Questions** (Need to test):
1. "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?" → Expected: "শুম্ভুনাথ"
2. "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?" → Expected: "মামাকে"
3. "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?" → Expected: "১৫ বছর"

**Status**: ❌ **Not Yet Verified** - Need to test these specific questions

### ✅ **BONUS TASKS - COMPLETED**

**Requirement**: Simple Conversation API
- ✅ **Implemented**: FastAPI REST API with multiple endpoints
- ✅ **Implemented**: `/query` endpoint for RAG interactions
- ✅ **Implemented**: `/health` endpoint for system monitoring
- ✅ **Implemented**: `/debug/retrieve` for debugging
- ✅ **Implemented**: Complete API documentation

**Requirement**: RAG Evaluation
- ✅ **Implemented**: Cosine similarity scores in retrieval
- ✅ **Implemented**: Relevance scoring in responses
- ✅ **Implemented**: Debug endpoints for evaluation
- ✅ **Implemented**: System health monitoring

### ✅ **INDUSTRY-STANDARD TOOLS - COMPLETED**

**✅ Our Tech Stack**:
- **Vector Database**: ChromaDB (industry standard)
- **LLM**: OpenAI GPT-3.5-turbo
- **Embeddings**: Cohere + OpenAI (dual approach)
- **Framework**: FastAPI + Streamlit
- **Containerization**: Docker + Docker Compose
- **Deployment**: AWS EC2 + nginx + Cloudflare

## 🚀 Features

- **Multilingual Support**: Handle queries in both English and Bengali
- **Dual Embeddings**: Uses both Cohere and OpenAI embeddings for better retrieval
- **OCR Processing**: Advanced PDF text extraction with OCR support
- **Vector Database**: ChromaDB for efficient similarity search
- **Reranking**: Cohere rerank for improved answer relevance
- **Web Interface**: Streamlit GUI for easy interaction
- **REST API**: FastAPI backend for programmatic access
- **Docker Support**: Complete containerization for easy deployment
- **Production Deployment**: AWS EC2 with nginx reverse proxy
- **Domain Configuration**: Ready for Cloudflare integration

## 📋 Prerequisites

- Docker and Docker Compose
- Python 3.9+
- API Keys:
  - Cohere API Key
  - OpenAI API Key
  - Pinecone API Key (optional)

## 🛠️ Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd multilingual-rag-system
```

### 2. Set Up Environment Variables
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Add Your PDF Documents
```bash
# Place your PDF files in the data/ directory
mkdir -p data
cp your_document.pdf data/
```

### 4. Start the System
```bash
# Start all services
make up

# Or start individual services
make api    # FastAPI backend only
make gui    # Streamlit GUI only
```

### 5. Ingest Your Documents
```bash
# Ingest a specific PDF
make ingest PDF_NAME=your_document.pdf

# Clear existing data and re-ingest
make ingest-clear PDF_NAME=your_document.pdf
```

### 6. Access the System
- **Web Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📁 Project Structure

```
├── docker-compose.yml          # Docker services configuration
├── Dockerfile                  # Main application Dockerfile
├── gui/                       # Streamlit GUI
│   ├── app.py                 # Main GUI application
│   ├── Dockerfile             # GUI Dockerfile
│   └── requirements.txt       # GUI dependencies
├── src/                       # Core application
│   ├── core/                  # Configuration and models
│   ├── services/              # Business logic services
│   ├── pipeline/              # Data processing pipelines
│   └── main.py               # FastAPI application
├── data/                      # PDF documents (mounted volume)
├── cache/                     # OCR cache (mounted volume)
├── Makefile                   # Development commands
└── requirements.txt           # Main application dependencies
```

## 🔧 Configuration

### Environment Variables
- `COHERE_API_KEY`: Your Cohere API key
- `OPENAI_API_KEY`: Your OpenAI API key
- `PINECONE_API_KEY`: Your Pinecone API key (optional)
- `EMBEDDING_PROVIDER`: "cohere", "openai", or "both"
- `TOP_K_RETRIEVAL`: Number of chunks to retrieve (default: 12)
- `RELEVANCE_THRESHOLD`: Minimum relevance score (default: 0.5)

### API Endpoints

#### Health Check
```bash
GET /health
```

#### Query Endpoint
```bash
POST /query
{
  "query": "Your question here",
  "top_k": 12
}
```

#### Debug Retrieval
```bash
POST /debug/retrieve
{
  "query": "Your question here",
  "top_k": 12
}
```

## 🐳 Docker Commands

```bash
# Start all services
make up

# Stop all services
make down

# View logs
make logs

# Rebuild images
make build

# Start only API
make api

# Start only GUI
make gui
```

## 📊 System Status

Check system health and vector database statistics:
```bash
curl http://localhost:8000/health
```

## 🔍 Troubleshooting

### Common Issues

1. **API Keys Not Set**
   - Ensure all required API keys are in your `.env` file

2. **PDF Processing Issues**
   - Check that PDF files are in the `data/` directory
   - Verify PDF files are not corrupted

3. **Vector Database Issues**
   - Clear and re-ingest data: `make ingest-clear PDF_NAME=your_file.pdf`

4. **OCR Processing Slow**
   - The system uses OCR caching to speed up re-processing
   - Check cache status in the GUI

### Debug Commands

```bash
# Check system health
curl http://localhost:8000/health

# Test API directly
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question here"}'
```

## 🎯 Assessment Questions & Answers

### What method or library did you use to extract the text, and why?
**Answer**: We used PyMuPDF (fitz) for basic text extraction and Tesseract OCR for Bengali text. We chose this combination because:
- PyMuPDF is fast for English text extraction
- Tesseract supports Bengali language with proper training
- OCR caching system prevents re-processing the same pages
- Parallel processing optimizes performance

### What chunking strategy did you choose?
**Answer**: We used RecursiveCharacterTextSplitter with Bengali-aware separators:
- Character limit: 1000 characters per chunk
- Overlap: 200 characters for context continuity
- Bengali separators: `।`, `?`, `!`, `\n\n`, `\n`, ` `, ``
- This works well for semantic retrieval because it preserves sentence boundaries while maintaining context

### What embedding model did you use?
**Answer**: We implemented a dual embedding approach:
- **Cohere**: `embed-multilingual-v3.0` for multilingual support
- **OpenAI**: `text-embedding-3-small` for high-quality embeddings
- This dual approach captures meaning better by leveraging both models' strengths

### How are you comparing the query with your stored chunks?
**Answer**: We use cosine similarity with ChromaDB:
- Dual embedding search (Cohere + OpenAI)
- Configurable top-k retrieval (default: 12)
- Cohere rerank for improved relevance
- ChromaDB provides efficient similarity search with metadata filtering

### How do you ensure meaningful comparison?
**Answer**: 
- Dual embedding approach captures different semantic aspects
- Reranking improves relevance of retrieved chunks
- Context-aware prompt engineering for answer generation
- For vague queries, the system retrieves multiple chunks and synthesizes an answer

### Do the results seem relevant?
**Answer**: The system shows good technical implementation but needs verification with the specific assessment questions. The dual embedding approach and reranking should provide relevant results, but Bengali OCR quality and chunking strategy may need optimization.

## 🚀 Production Deployment

### AWS EC2 Deployment
- ✅ **Deployed**: Complete system on AWS EC2 instance
- ✅ **Configured**: nginx reverse proxy
- ✅ **Secured**: API internal access, GUI public access
- ✅ **Domain Ready**: Configured for `ragbangla.nimbusrb.com`

### Deployment Commands
```bash
# SSH to EC2 instance
ssh -i your-key.pem ec2-user@your-ec2-ip

# Start services
cd /path/to/project
docker-compose up -d

# Check status
docker-compose ps
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs: `make logs`
3. Open an issue on GitHub

## ⚠️ Known Limitations

1. **Bengali OCR Quality**: May need improvement for complex Bengali text
2. **Assessment Question Testing**: Need to verify specific Bengali question accuracy
3. **Chunking Optimization**: May need fine-tuning for Bengali content
4. **Rate Limiting**: Cohere trial tier has rate limits affecting ingestion speed 