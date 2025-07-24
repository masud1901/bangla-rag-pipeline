# Multilingual RAG System

A Retrieval-Augmented Generation (RAG) system that supports both English and Bengali queries, built with FastAPI, Streamlit, and Docker.

## 🚀 Features

- **Multilingual Support**: Handle queries in both English and Bengali
- **Dual Embeddings**: Uses both Cohere and OpenAI embeddings for better retrieval
- **OCR Processing**: Advanced PDF text extraction with OCR support
- **Vector Database**: ChromaDB for efficient similarity search
- **Reranking**: Cohere rerank for improved answer relevance
- **Web Interface**: Streamlit GUI for easy interaction
- **REST API**: FastAPI backend for programmatic access
- **Docker Support**: Complete containerization for easy deployment

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