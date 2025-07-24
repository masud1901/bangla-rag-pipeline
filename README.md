# Multilingual RAG System - RAG pipeline for HSC Bangla Text Book

This project is my attempt to build a Retrieval-Augmented Generation (RAG) system that can understand both English and Bengali. I built it using FastAPI, Streamlit, and Docker as part of the AI Engineer (Level-1) Technical Assessment. The goal was to create a tool that could answer questions based on the content of a Bengali textbook.

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

## 🚀 What This System Can Do

- **Works in English & Bengali**: The system can handle questions in two languages.
- **Dual Embedding System**: I used both Cohere and OpenAI models to better understand the text.
- **PDF Text Extraction with OCR**: It can read text from PDFs, using OCR for scanned pages.
- **Vector Search with ChromaDB**: Uses ChromaDB to find relevant information quickly.
- **Improved Relevance with Reranking**: A reranking step helps make sure the retrieved information is accurate.
- **Simple Web Interface**: A Streamlit app provides a simple UI for asking questions.
- **REST API**: A FastAPI backend allows other programs to connect to it.
- **Dockerized**: The whole system is packaged with Docker for easier setup.
- **Deployed on AWS**: The project is live on an AWS EC2 instance using Nginx.

## 📋 Prerequisites

- Docker and Docker Compose
- Python 3.9+
- API Keys:
  - Cohere API Key
  - OpenAI API Key

## 🛠️ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/masud1901/bangla-rag-pipeline
cd bangla-rag-pipeline
```

### 2. Set Up Environment Variables
```bash
# Create a .env file from the example
cp .env.example .env

# Edit .env and add your API keys
# COHERE_API_KEY=...
# OPENAI_API_KEY=...
```

### 3. Add Your PDF Documents
```bash
# Place your PDF files in the data/ directory
mkdir -p data
cp your_document.pdf data/
```

### 4. Start the System
```bash
# Start all services in the background
make up
```

### 5. Ingest Your Documents
```bash
# Ingest a specific PDF file from the `data` directory
make ingest PDF_NAME=your_document.pdf

# To clear the database and start fresh
make ingest-clear PDF_NAME=your_document.pdf
```

### 6. Access the System
- **Live System**: [https://ragbangla.nimbusrb.com](https://ragbangla.nimbusrb.com)
- **Local Web Interface**: [http://localhost:8503](http://localhost:8503)
- **API Documentation**: [http://localhost:8003/docs](http://localhost:8003/docs)
- **Health Check**: [http://localhost:8003/health](http://localhost:8003/health)

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
├── chroma_db/                 # Persistent vector database (mounted volume)
├── Makefile                   # Development commands
└── requirements.txt           # Main application dependencies
```

## 🔧 Configuration

### Environment Variables
- `COHERE_API_KEY`: Your Cohere API key.
- `OPENAI_API_KEY`: Your OpenAI API key.
- `EMBEDDING_PROVIDER`: "cohere", "openai", or "both".
- `TOP_K_RETRIEVAL`: Number of chunks to retrieve (default: 12).
- `RELEVANCE_THRESHOLD`: Minimum relevance score (default: 0.5).

### API Endpoints

#### Health Check
`GET /health`

#### Query Endpoint
`POST /query`
```json
{
  "query": "Your question here",
  "top_k": 12
}
```

## 🎯 Assessment Questions & Answers

### What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?
**Answer**: I implemented a robust, hybrid text extraction strategy using **PyMuPDF (`fitz`)** for digitally native text and **Tesseract OCR** for image-based or scanned text.

- **Why this approach?**
    - **PyMuPDF** is exceptionally fast and accurate for extracting text directly embedded within a PDF. It served as my primary, high-speed extractor.
    - **Tesseract OCR** was my essential fallback for pages that returned little or no text from PyMuPDF, indicating they were images. Tesseract has strong support for the Bengali language (`ben`), which was critical for processing the "HSC Bangla 1st paper" document.
    - To optimize performance, I implemented **parallel processing** for the OCR step and **cached the results**, making subsequent ingestions of the same document almost instantaneous.

- **Formatting Challenges Faced**:
    - **Mixed Content**: The primary PDF contained both digital text and scanned pages, making a single extraction method insufficient. My hybrid approach solved this automatically.
    - **Two-Column Layouts**: Some pages featured two-column text, which can disrupt reading order. While my current method extracts text linearly, it proved sufficient for good retrieval.
    - **OCR Inaccuracies**: OCR on Bengali text, especially with older prints, sometimes struggled with complex conjunct characters (যুক্তবর্ণ). This is a known challenge that my system mitigates but highlights an area for future improvement.

### What chunking strategy did you choose and why do you think it works well for semantic retrieval?
**Answer**: I chose the **`RecursiveCharacterTextSplitter`**, a sophisticated strategy that preserves the semantic integrity of the text.

- **Why it's effective for semantic retrieval**:
    - **Maintains Cohesion**: This method attempts to split text along a prioritized list of separators, starting with the most semantically significant ones. For my Bengali content, I defined a custom separator list: `["।", "?", "!", "\n\n", "\n", " "]`. This means the splitter tries to keep full paragraphs and sentences intact before resorting to smaller splits, which is crucial for generating meaningful embeddings.
    - **Optimal Chunk Size**: I configured a `chunk_size` of 1000 characters. This is a balance—large enough to contain a complete idea but small enough to provide a focused, low-noise context for the LLM.
    - **Contextual Overlap**: I set a `chunk_overlap` of 200 characters. This creates a "sliding window" between chunks, ensuring that concepts spanning a chunk boundary are fully captured in at least one of the chunks, preventing context loss.

By creating chunks that are semantically whole, I generate more accurate vector representations, leading directly to more relevant and precise search results.

### What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?
**Answer**: I implemented a powerful **dual-embedding strategy**, leveraging two state-of-the-art models simultaneously: **Cohere's `embed-multilingual-v3.0`** and **OpenAI's `text-embedding-3-small`**.

- **Why a dual-model approach?**
    - **Cohere `embed-multilingual-v3.0`**: This model was specifically chosen for its exceptional performance in over 100 languages, including Bengali. It excels at creating a shared "meaning space" where similar concepts in different languages are mapped to nearby vectors, which is perfect for my core multilingual requirement.
    - **OpenAI `text-embedding-3-small`**: This is a highly performant and efficient model that provides a robust, general-purpose understanding of text. I included it to capture nuances and to provide a complementary perspective on the text's meaning.

- **How it works**:
    During ingestion, every chunk is embedded using both models and stored in separate collections in my ChromaDB vector store. During a query, the user's question is also embedded twice, and I search both collections. This ensemble approach increases the chances of finding the most relevant content by drawing from the unique strengths of two different world-class models.

### How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?
**Answer**: My retrieval process is a multi-stage pipeline designed for high precision, using **ChromaDB** for storage, **Cosine Similarity** for the initial search, and **Cohere's Rerank** for refinement.

- **Storage (ChromaDB)**: I chose ChromaDB because it's a lightweight, developer-friendly, and open-source vector database that is easy to containerize and deploy. Its file-based persistence is perfect for my project, ensuring embeddings are saved across sessions.
- **Similarity Search (Cosine Similarity)**: This is the standard for comparing high-dimensional text embeddings. It measures the angle between two vectors, focusing purely on their semantic direction rather than their magnitude. This makes it excellent at identifying chunks with similar *meaning*, regardless of their length.
- **Refinement (Cohere Rerank)**: This is my critical final step. After retrieving an initial set of chunks with cosine similarity, I pass them to Cohere's Rerank endpoint. This is a more advanced model (a cross-encoder) that directly compares the query text against each chunk to produce a highly accurate relevance score. This significantly improves the final quality of the context sent to the LLM, ensuring only the most relevant information is used to generate the answer.

### How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?
**Answer**: I ensure meaningful comparison through my multi-layered retrieval and generation strategy.

- **Ensuring Meaningful Comparison**:
    1.  **Dual-Embedding Ensemble**: Using two distinct, high-quality embedding models provides a more robust semantic match than relying on a single model's interpretation.
    2.  **Sophisticated Reranking**: The Cohere Rerank model acts as a powerful judge, re-ordering the initial search results based on a deeper, contextual comparison between the query and each chunk. This is my primary mechanism for ensuring relevance.
    3.  **Context-Aware Prompting**: The final prompt sent to the LLM is engineered to instruct it to synthesize an answer *exclusively* from the provided, reranked context, which minimizes hallucinations.

- **Handling Vague or Context-Free Queries**:
    My system is designed to handle ambiguity gracefully. If a query like "tell me more" is entered:
    - The initial vector search might retrieve a broad set of thematically related chunks.
    - The reranking model will attempt to find the most relevant chunks from this set, but the relevance scores may be low.
    - The LLM, seeing a diverse and potentially unfocused context, is instructed to either synthesize a general summary based on the available information or, if the context is completely irrelevant, to state that it cannot provide a specific answer from the documents. This prevents the model from inventing information.

## 🚀 Production Deployment

### AWS EC2 Deployment
- ✅ **Deployed**: Complete system on an AWS EC2 instance.
- ✅ **Configured**: Nginx is set up as a reverse proxy to manage traffic.
- ✅ **Secured**: The API is only internally accessible, while the GUI is exposed to the public.
- ✅ **Domain Ready**: The system is live and configured for the domain [**ragbangla.nimbusrb.com**](https://ragbangla.nimbusrb.com).

## ⚠️ Current Limitations

1.  **Bengali OCR is not perfect**: The current OCR system struggles with some Bengali text, which is the biggest issue affecting performance.
2.  **Formal Testing Needed**: The sample questions from the assessment haven't been formally checked for accuracy.
3.  **Chunking could be better**: The way the text is split into chunks works, but it could probably be improved.
4.  **API Rate Limits**: If you use a free API key from Cohere, the initial data processing can be slow.

## 📝 Project Reflection & Future Plans

### Building a Solid System, But Hitting a Real-World Roadblock
This project gave me the chance to build a complete Retrieval-Augmented Generation (RAG) pipeline. On the technical side, it was a solid achievement. I used a dual-embedding strategy, advanced reranking, Docker for containerization, and successfully deployed it on AWS EC2. These are all valuable skills, and I’m proud of what I’ve learned and built.

However, while the system worked well technically, the results—especially for Bengali queries—were not good enough. This made me realize an important lesson: **even a well-designed system can fail if the data it works with is poor**.

### What Went Wrong: OCR Was the Weak Link
After checking everything step by step, I found that the main issue came from the **OCR (Optical Character Recognition)** system. I used Tesseract to read Bengali text, but it didn’t perform well. It often produced incorrect or messy text. Since this faulty text was used to create embeddings, the entire search and response system was affected.

Because of this, I couldn’t even properly evaluate the performance of the embedding and reranking models. The base data was already too noisy. So, before moving forward, **fixing the OCR step is the top priority**.

### Looking Ahead: A Smarter Way to Support Students
Even though this part of the work ends here, I see a lot of potential in this system. My long-term goal is to turn it into an **AI-powered tutor**—a helpful tool that can answer student questions based on textbook content, especially in Bengali. This could be a great way to make learning easier and more accessible, especially in areas where good tutoring isn’t available.

The current system is already a strong base. With better input data, I believe it can grow into something truly helpful.

### What’s Next: Small Steps, Clear Goals
To move this forward, I need to take some practical steps:

1. **Find a better OCR solution** for Bengali text, possibly using newer models or even training one with custom data.
2. **Talk to students and teachers** to understand if this tool would actually be useful to them.
3. Keep improving the pipeline step by step, making sure it stays focused on real-world impact—not just technical performance.

### Final Thoughts
This project taught me that technical success doesn’t always mean real-world success. Good engineering needs both strong systems **and** reliable data. I’m grateful for what I’ve learned here—and more motivated than ever to build tools that actually help people. 