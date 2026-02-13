"""Multi-Index RAG avec 4 indexes spécialisés."""

import logging
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from .hierarchical_chunker import HierarchicalChunk
from .config import config

EMBEDDING_BATCH_SIZE = 100  # Batch pour éviter limite 300k tokens OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiIndexRAG:
    """
    RAG Multi-Index avec 4 indexes spécialisés:
    1. Chapter Index - chunks de chapitre (contexte large)
    2. Section Index - chunks de section (contexte moyen)
    3. Paragraph Index - chunks de paragraphe (précision)
    4. Full Index - tous les chunks (recherche générale)
    """

    def __init__(self, embeddings):
        """
        Initialiser le Multi-Index RAG.

        Args:
            embeddings: Modèle d'embeddings (OpenAI ou HuggingFace)
        """
        self.embeddings = embeddings

        # 4 indexes FAISS spécialisés
        self.chapter_index: Optional[FAISS] = None
        self.section_index: Optional[FAISS] = None
        self.paragraph_index: Optional[FAISS] = None
        self.full_index: Optional[FAISS] = None

        # Documents pour BM25
        self.chapter_docs: List[Document] = []
        self.section_docs: List[Document] = []
        self.paragraph_docs: List[Document] = []
        self.full_docs: List[Document] = []

        # Retrievers hybrides
        self.chapter_retriever = None
        self.section_retriever = None
        self.paragraph_retriever = None
        self.full_retriever = None

    def build_indexes(
        self,
        hierarchical_chunks: List[HierarchicalChunk],
        documents: List[Document]
    ):
        """
        Construire les 4 indexes à partir des chunks hiérarchiques.

        Args:
            hierarchical_chunks: Chunks hiérarchiques
            documents: Documents LangChain correspondants
        """
        logger.info("Construction des 4 indexes spécialisés...")

        # Séparer les documents par niveau
        chapter_docs = [doc for doc in documents if doc.metadata.get("level") == 1]
        section_docs = [doc for doc in documents if doc.metadata.get("level") == 2]
        paragraph_docs = [doc for doc in documents if doc.metadata.get("level") == 3]

        # Stocker pour BM25
        self.chapter_docs = chapter_docs
        self.section_docs = section_docs
        self.paragraph_docs = paragraph_docs
        self.full_docs = documents

        # 1. Chapter Index (contexte large)
        if chapter_docs:
            logger.info(f"Création Chapter Index ({len(chapter_docs)} chunks)...")
            self.chapter_index = self._create_faiss_batched(chapter_docs)
            self.chapter_retriever = self._create_hybrid_retriever(
                self.chapter_index,
                chapter_docs,
                "chapter"
            )

        # 2. Section Index (contexte moyen)
        if section_docs:
            logger.info(f"Création Section Index ({len(section_docs)} chunks)...")
            self.section_index = self._create_faiss_batched(section_docs)
            self.section_retriever = self._create_hybrid_retriever(
                self.section_index,
                section_docs,
                "section"
            )

        # 3. Paragraph Index (précision)
        if paragraph_docs:
            logger.info(f"Création Paragraph Index ({len(paragraph_docs)} chunks)...")
            self.paragraph_index = self._create_faiss_batched(paragraph_docs)
            self.paragraph_retriever = self._create_hybrid_retriever(
                self.paragraph_index,
                paragraph_docs,
                "paragraph"
            )

        # 4. Full Index (tous les niveaux)
        logger.info(f"Création Full Index ({len(documents)} chunks)...")
        self.full_index = self._create_faiss_batched(documents)
        self.full_retriever = self._create_hybrid_retriever(
            self.full_index,
            documents,
            "full"
        )

        logger.info("✅ 4 indexes créés avec succès")

    def _create_faiss_batched(self, documents: List[Document]) -> FAISS:
        """Créer FAISS par batch pour éviter la limite de tokens OpenAI."""
        batch_size = EMBEDDING_BATCH_SIZE

        if len(documents) <= batch_size:
            return FAISS.from_documents(documents, self.embeddings)

        logger.info(f"  FAISS batch: {len(documents)} docs, batch_size={batch_size}")
        vector_store = FAISS.from_documents(documents[:batch_size], self.embeddings)

        for i in range(batch_size, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vector_store.add_documents(batch)

        return vector_store

    def _create_hybrid_retriever(
        self,
        vector_store: FAISS,
        documents: List[Document],
        index_name: str
    ) -> EnsembleRetriever:
        """
        Créer un retriever hybride (MMR + BM25) pour un index.

        Args:
            vector_store: FAISS vector store
            documents: Documents pour BM25
            index_name: Nom de l'index

        Returns:
            Retriever hybride
        """
        if not config.USE_HYBRID_SEARCH:
            # Fallback: MMR only
            return vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": config.TOP_K_RESULTS,
                    "fetch_k": config.TOP_K_RESULTS * config.MMR_FETCH_K_MULTIPLIER,
                    "lambda_mult": config.MMR_LAMBDA,
                }
            )

        # FAISS avec MMR
        faiss_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": config.TOP_K_RESULTS,
                "fetch_k": config.TOP_K_RESULTS * config.MMR_FETCH_K_MULTIPLIER,
                "lambda_mult": config.MMR_LAMBDA,
            }
        )

        # BM25
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = config.TOP_K_RESULTS

        # Ensemble
        hybrid_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[config.FAISS_WEIGHT, config.BM25_WEIGHT],
        )

        logger.info(f"Retriever hybride créé pour {index_name}")
        return hybrid_retriever

    def get_retriever(self, index_type: str = "full"):
        """
        Obtenir un retriever spécifique.

        Args:
            index_type: "chapter", "section", "paragraph", ou "full"

        Returns:
            Retriever correspondant
        """
        retrievers = {
            "chapter": self.chapter_retriever,
            "section": self.section_retriever,
            "paragraph": self.paragraph_retriever,
            "full": self.full_retriever,
        }

        retriever = retrievers.get(index_type)
        if not retriever:
            logger.warning(f"Index {index_type} non disponible, fallback sur full")
            return self.full_retriever

        return retriever

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Obtenir les statistiques des indexes.

        Returns:
            Statistiques par index
        """
        return {
            "chapter_index": {
                "exists": self.chapter_index is not None,
                "chunks": len(self.chapter_docs),
            },
            "section_index": {
                "exists": self.section_index is not None,
                "chunks": len(self.section_docs),
            },
            "paragraph_index": {
                "exists": self.paragraph_index is not None,
                "chunks": len(self.paragraph_docs),
            },
            "full_index": {
                "exists": self.full_index is not None,
                "chunks": len(self.full_docs),
            },
        }

    def reset(self):
        """Réinitialiser tous les indexes."""
        logger.info("Réinitialisation des indexes...")
        self.chapter_index = None
        self.section_index = None
        self.paragraph_index = None
        self.full_index = None
        self.chapter_docs = []
        self.section_docs = []
        self.paragraph_docs = []
        self.full_docs = []
        self.chapter_retriever = None
        self.section_retriever = None
        self.paragraph_retriever = None
        self.full_retriever = None
