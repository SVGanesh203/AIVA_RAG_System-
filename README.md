# KB-Bot: RAG System with HuggingFace Embeddings & Ollama LLM

A comprehensive Knowledge Base Bot that leverages RAG (Retrieval Augmented Generation) with HuggingFace embeddings and Ollama local LLM for intelligent document search and Q&A.

## Features

- **🧠 Semantic Search**: Uses HuggingFace sentence-transformers for intelligent document retrieval
- **📄 Multi-Format Support**: Process PDF and PowerPoint documents
- **🤖 Local LLM Integration**: Powered by Ollama for privacy-first responses
- **🎯 Intelligent Chunking**: RecursiveCharacterTextSplitter with context preservation
- **🔐 Authentication**: Session-based user authentication
- **💾 Vector Storage**: FAISS-based document indexing
- **🎨 Web Interface**: Flask-based interactive UI

## Architecture

### Core Components

```
KB-Bot/
├── app.py                          # Flask application entry point
├── config.py                       # Configuration & dependencies
├── rag_system.py                  # RAG query interface & context management
├── unified_document_processor.py  # Document processing pipeline
├── routes/
│   ├── auth_routes.py            # Authentication endpoints
│   ├── chat_routes.py            # Chat/Q&A endpoints
│   ├── document_routes.py        # Document upload/management
│   ├── embedding_routes.py       # Embedding generation
│   └── system_routes.py          # System info endpoints
├── modules/
│   ├── document_manager.py       # Document loading & FAISS operations
│   ├── conversation.py           # Conversation context management
│   ├── auth.py                   # Authentication logic
│   ├── search.py                 # Search utilities
│   ├── response_generator.py     # LLM response generation
│   └── utils.py                  # Helper functions
├── embedding/                     # Document vector storage
├── templates/                     # HTML templates
└── static/                        # CSS, JS, assets
```

## Technology Stack

### Embeddings & NLP
- **HuggingFace Sentence-Transformers**: `all-mpnet-base-v2` (primary), `all-MiniLM-L6-v2` (fallback)
- **Transformers**: `5.2.0`
- **Sentence-Transformers**: `5.2.3`
- **Tokenizers**: `0.22.2`

### Vector Database
- **FAISS** (Facebook AI Similarity Search) - Fast semantic search

### Language Model
- **Ollama**: Local LLM engine (supports models like tinyllama, llama2, etc.)

### Web Framework
- **Flask**: Lightweight Python web framework
- **Flask-Session**: Session management

### Document Processing
- **LangChain**: Document loading and text splitting
  - `langchain-community`: Document loaders
  - `langchain-core`: Core abstractions
  - `langchain-text-splitters`: RecursiveCharacterTextSplitter
  - `langchain-huggingface`: HuggingFace embeddings
- **PyPDF, PDFPlumber**: PDF extraction
- **python-pptx**: PowerPoint processing
- **EasyOCR**: Optical character recognition
- **PyMuPDF**: Advanced PDF analysis

### ML/AI
- **PyTorch**: Deep learning framework
- **scikit-learn**: Machine learning utilities
- **NumPy, SciPy**: Numerical computing

## Installation

### Prerequisites
- Python 3.8+
- Ollama running locally ([Download](https://ollama.ai))
- Git

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/SVGanesh203/kb-bot.git
cd kb-bot
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables** (optional)
```bash
cp .env.example .env
# Edit .env with your settings
```

5. **Pull LLM model with Ollama**
```bash
ollama pull tinyllama  # Or your preferred model
```

6. **Run the application**
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Configuration

### Key Settings in `config.py`

```python
# Embedding Models (tried in order)
EMBED_MODEL_OPTIONS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",      # Default
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
]

# Document Processing
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MAX_FILE_SIZE_MB = 100

# Search Behavior
FAST_QUERY_K = 12
MAX_CONTEXT_CHARS = 10000
MIN_RELEVANCE_THRESHOLD = 0.3

# Device (auto-detected)
DEVICE = 'cuda' | 'mps' | 'cpu'  # Automatically selected
```

## API Endpoints

### Authentication
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout
- `POST /api/auth/register` - User registration

### Chat
- `POST /api/chat` - Send question and get answer
- `GET /api/chat/history` - Get conversation history

### Documents
- `POST /api/documents/upload` - Upload PDF/PPT
- `GET /api/documents` - List uploaded documents
- `DELETE /api/documents/{doc_id}` - Delete document

### System
- `GET /api/health` - Health check
- `GET /api/system/info` - System information

## Usage Example

### Upload a Document
```bash
curl -X POST -F "file=@document.pdf" http://localhost:5000/api/documents/upload
```

### Ask a Question
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?"}' \
  http://localhost:5000/api/chat
```

## How HuggingFace Models Are Used

1. **Document Indexing**: Documents → Split into chunks → Convert to embeddings using HuggingFace model → Store in FAISS
2. **Query Processing**: User question → Convert to embedding using same model → FAISS similarity search → Retrieve relevant chunks
3. **Answer Generation**: Retrieved chunks + query → Send to Ollama LLM → Generate contextual answer

## Performance Optimization

- **Device Selection**: Automatically uses GPU (CUDA/MPS) or falls back to CPU
- **Vector Caching**: Caches recent search results
- **Batch Processing**: Processes documents in batches for memory efficiency
- **Parallel Search**: Multi-threaded document searching

## Troubleshooting

### Ollama Connection Failed
```bash
# Ensure Ollama is running
ollama serve

# Or pull a model
ollama pull tinyllama
```

### Out of Memory
- Reduce `CHUNK_SIZE` in config.py
- Use smaller embedding model: `all-MiniLM-L6-v2`
- Set device to 'cpu' in .env

### Slow Embedding
- Use `all-MiniLM-L6-v2` instead of `all-mpnet-base-v2`
- Enable GPU acceleration (CUDA/MPS)

## Project Structure

```
requirements.txt        # Python dependencies
config.py              # Configuration & setup
app.py                 # Main Flask app
rag_system.py          # RAG core logic
unified_document_processor.py  # Document handling
app_state.py           # Global app state
init_check.py          # Initialization checks
```

## Dependencies Highlights

| Package | Purpose |
|---------|---------|
| `sentence-transformers` | Semantic embeddings |
| `langchain-*` | Document processing & RAG |
| `faiss-cpu` | Vector similarity search |
| `ollama` | Local language model |
| `flask` | Web framework |
| `torch` | Deep learning |
| `pytesseract/easyocr` | Document OCR |

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Debug Mode
```bash
export FLASK_ENV=development
python app.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

SVGanesh203 - [GitHub](https://github.com/SVGanesh203)

## Support

For issues and feature requests, please use the [GitHub Issues](https://github.com/SVGanesh203/kb-bot/issues) page.

---

**Note**: This is a local-first RAG system prioritizing privacy and offline capability. All processing happens on your machine with no external API calls needed (after model download).
