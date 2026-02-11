# HAL Science RAG Chatbot 🔬

A specialized Retrieval-Augmented Generation (RAG) chatbot for scientific research using the [HAL Science open archives](https://hal.science). Search for scientific theses and engage in intelligent conversations with document content.

## Features

- **🔍 Discovery Phase**: Search and filter theses from HAL Science archives
- **💬 Deep Dive Phase**: Chat with selected documents using advanced RAG pipeline
- **📄 Smart PDF Processing**: Automatic text extraction, chunking, and vectorization
- **🎯 Source Citations**: View relevant document chunks supporting each answer
- **⚡ Optimized Performance**: FAISS vector store for fast similarity search
- **🔧 Configurable**: Support for OpenAI or HuggingFace embeddings

## Architecture

```
HAL API → PDF Download → Text Extraction → Chunking → Embeddings → Vector Store → LLM Chat
```

### Tech Stack

- **Frontend**: Streamlit
- **RAG Framework**: LangChain
- **Vector Store**: FAISS
- **PDF Processing**: PyMuPDF (fitz)
- **Embeddings**: OpenAI or HuggingFace (sentence-transformers)
- **LLM**: OpenAI GPT-4

## Installation

### 1. Clone and Setup

```bash
cd hal-rag-chatbot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-key-here
```

**Note**: OpenAI API key is required for full functionality (embeddings + LLM). Alternatively, you can use free HuggingFace embeddings for the vector store, but you'll need to configure a local LLM separately.

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Phase 1: Search & Discovery

1. Enter research keywords (e.g., "machine learning", "climate change")
2. Browse search results with titles, abstracts, authors
3. Select a thesis with available PDF to proceed

### Phase 2: Chat with Document

1. Wait for PDF processing (automatic chunking and vectorization)
2. Ask questions about the document content
3. Receive AI-generated answers with source citations
4. View relevant document chunks supporting each answer

## Configuration Options

Edit `.env` file to customize:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_MODEL` | GPT model for chat | `gpt-4-turbo-preview` |
| `CHUNK_SIZE` | Text chunk size (characters) | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `TOP_K_RESULTS` | Number of relevant chunks to retrieve | `4` |
| `USE_HUGGINGFACE_EMBEDDINGS` | Use free HuggingFace instead of OpenAI | `False` |
| `VECTOR_STORE_TYPE` | Vector store backend | `faiss` |

## Project Structure

```
hal-rag-chatbot/
├── app.py                 # Streamlit main application
├── src/
│   ├── api_client.py      # HAL API wrapper
│   ├── rag_engine.py      # RAG pipeline (PDF→Vector Store→Chat)
│   ├── config.py          # Configuration management
│   └── utils.py           # Helper functions
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
└── README.md              # This file
```

## Key Components

### HAL API Client (`src/api_client.py`)

- Search HAL archives with filtering for theses
- Handle pagination and error recovery
- Download PDFs from HAL servers
- Parse metadata (title, author, abstract, keywords)

### RAG Engine (`src/rag_engine.py`)

- **PDF Processing**: Extract and clean text from PDFs
- **Text Chunking**: RecursiveCharacterTextSplitter with overlap
- **Vectorization**: Create embeddings using OpenAI or HuggingFace
- **Vector Store**: FAISS for efficient similarity search
- **Conversational Chain**: LangChain ConversationalRetrievalChain
- **Memory**: Maintain chat history and context

## API Reference

### HAL API Endpoints

The application uses the HAL Science API:
- **Base URL**: `https://api.archives-ouvertes.fr/search/`
- **Filter**: `docType_s:THESE` (theses only)
- **Documentation**: https://api.archives-ouvertes.fr/docs

### Example API Query

```python
from src.api_client import HALAPIClient

client = HALAPIClient()
results = client.search_theses(query="machine learning", rows=10)

for doc in results["docs"]:
    print(f"{doc.title} by {doc.author}")
```

## Troubleshooting

### No OpenAI API Key

If you don't have an OpenAI API key:
1. Set `USE_HUGGINGFACE_EMBEDDINGS=True` in `.env`
2. This enables free embeddings but requires additional LLM setup
3. Consider using Ollama or other local LLMs

### PDF Download Fails

- Some PDFs may be restricted or behind authentication
- Check the HAL URL directly in a browser
- Try a different thesis with available PDF

### Memory Issues

For large PDFs:
- Reduce `CHUNK_SIZE` in `.env`
- Reduce `TOP_K_RESULTS` to retrieve fewer chunks
- Increase available system RAM

### Slow Performance

- Use `gpt-3.5-turbo` instead of `gpt-4` for faster responses
- Reduce number of chunks (`TOP_K_RESULTS`)
- Use FAISS instead of ChromaDB for vector store

## Roadmap

- [ ] Support for multiple document comparison
- [ ] Export chat history to PDF/Markdown
- [ ] Advanced filters (date range, domain, institution)
- [ ] Citation generation in various formats
- [ ] Multi-language support
- [ ] Integration with other scientific archives (arXiv, PubMed)

## Contributing

Contributions are welcome! Areas for improvement:
- Better PDF parsing for complex layouts
- Support for additional vector stores (Pinecone, Weaviate)
- Enhanced UI/UX in Streamlit
- Unit tests and integration tests
- Docker containerization

## License

MIT License - feel free to use for research and education.

## Acknowledgments

- [HAL Science](https://hal.science) for providing open access to scientific research
- [LangChain](https://langchain.com) for the RAG framework
- [OpenAI](https://openai.com) for embeddings and LLM
- [Streamlit](https://streamlit.io) for the web framework

## Contact

For questions or issues, please open a GitHub issue or contact the maintainer.

---

**Happy researching!** 🔬📚
