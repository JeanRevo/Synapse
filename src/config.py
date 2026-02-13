"""Gestion de la configuration du Chatbot RAG HAL."""

import os
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class Config(BaseModel):
    """Configuration de l'application."""

    # Configuration de l'API HAL
    HAL_API_BASE_URL: str = "https://api.archives-ouvertes.fr/search/"
    HAL_API_TIMEOUT: int = 30
    HAL_RESULTS_PER_PAGE: int = 10

    # Configuration OpenAI
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Alternative: Embeddings HuggingFace (si OpenAI n'est pas utilisé)
    USE_HUGGINGFACE_EMBEDDINGS: bool = os.getenv("USE_HUGGINGFACE_EMBEDDINGS", "False").lower() == "true"
    HUGGINGFACE_MODEL: str = os.getenv("HUGGINGFACE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Configuration RAG
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "8"))

    # Configuration RAG Avancé (Phase 1)
    USE_HYBRID_SEARCH: bool = os.getenv("USE_HYBRID_SEARCH", "True").lower() == "true"
    FAISS_WEIGHT: float = float(os.getenv("FAISS_WEIGHT", "0.6"))  # Poids vectoriel
    BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.4"))  # Poids mots-clés
    MMR_LAMBDA: float = float(os.getenv("MMR_LAMBDA", "0.7"))  # Balance pertinence/diversité
    MMR_FETCH_K_MULTIPLIER: int = int(os.getenv("MMR_FETCH_K_MULTIPLIER", "3"))  # fetch_k = k * multiplier

    # Configuration RAG Avancé (Phase 2)
    USE_HIERARCHICAL_CHUNKING: bool = os.getenv("USE_HIERARCHICAL_CHUNKING", "True").lower() == "true"
    USE_MULTI_INDEX: bool = os.getenv("USE_MULTI_INDEX", "True").lower() == "true"
    CHAPTER_CHUNK_SIZE: int = int(os.getenv("CHAPTER_CHUNK_SIZE", "1500"))
    SECTION_CHUNK_SIZE: int = int(os.getenv("SECTION_CHUNK_SIZE", "1000"))
    PARAGRAPH_CHUNK_SIZE: int = int(os.getenv("PARAGRAPH_CHUNK_SIZE", "500"))

    # Base de données vectorielle
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "faiss")  # faiss ou chromadb

    # Traitement des PDF
    PDF_DOWNLOAD_TIMEOUT: int = 120
    MAX_PDF_SIZE_MB: int = 50

    # Application
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Fonctionnalités ML
    ENABLE_CLASSIFICATION: bool = os.getenv("ENABLE_CLASSIFICATION", "True").lower() == "true"
    ENABLE_SUMMARIZATION: bool = os.getenv("ENABLE_SUMMARIZATION", "True").lower() == "true"
    ENABLE_RECOMMENDATIONS: bool = os.getenv("ENABLE_RECOMMENDATIONS", "True").lower() == "true"
    USE_ML_MODELS: bool = os.getenv("USE_ML_MODELS", "False").lower() == "true"

    class Config:
        """Configuration Pydantic."""
        env_file = ".env"
        case_sensitive = True


# Instance globale de configuration
config = Config()
