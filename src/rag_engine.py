"""Moteur RAG pour le traitement PDF et les réponses aux questions."""

import io
import logging
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from .config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HuggingFaceEmbeddings:
    """Wrapper personnalisé d'embeddings HuggingFace pour la compatibilité LangChain."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialiser les embeddings HuggingFace."""
        logger.info(f"Loading HuggingFace model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Encoder une liste de documents."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Encoder une requête unique."""
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()


class PDFProcessor:
    """Gérer l'analyse et l'extraction de texte des PDF."""

    @staticmethod
    def extract_text_from_pdf(pdf_bytes: bytes) -> str:
        """
        Extraire le texte des bytes PDF en utilisant PyMuPDF.

        Args:
            pdf_bytes: Contenu du fichier PDF en bytes

        Returns:
            Texte extrait sous forme de chaîne
        """
        try:
            logger.info("Starting PDF text extraction")
            pdf_stream = io.BytesIO(pdf_bytes)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")

            text_content = []
            total_pages = len(doc)

            for page_num in range(total_pages):
                page = doc[page_num]
                text = page.get_text("text")

                # Nettoyage basique
                text = text.strip()
                if text:
                    text_content.append(text)

                if page_num % 10 == 0:
                    logger.info(f"Processed {page_num + 1}/{total_pages} pages")

            doc.close()
            full_text = "\n\n".join(text_content)

            logger.info(f"Extracted {len(full_text)} characters from {total_pages} pages")
            return full_text

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Nettoyer le texte extrait.

        Args:
            text: Texte brut du PDF

        Returns:
            Texte nettoyé
        """
        # Supprimer les espaces excessifs
        text = " ".join(text.split())

        # Supprimer les numéros de page (heuristique simple)
        lines = text.split("\n")
        cleaned_lines = [line for line in lines if not (line.strip().isdigit() and len(line.strip()) < 4)]

        return "\n".join(cleaned_lines)


class RAGEngine:
    """Moteur RAG pour l'ingestion de documents et les réponses aux questions."""

    def __init__(self):
        """Initialiser le moteur RAG."""
        self.vector_store: Optional[FAISS] = None
        self.conversation_chain: Optional[ConversationalRetrievalChain] = None
        self.chat_history: List[Tuple[str, str]] = []
        self.current_doc_id: Optional[str] = None
        self.embeddings = None
        self.llm = None

        self._initialize_models()

    def _initialize_models(self):
        """Initialiser les modèles d'embeddings et LLM."""
        try:
            # Initialiser les embeddings
            if config.USE_HUGGINGFACE_EMBEDDINGS:
                logger.info("Using HuggingFace embeddings")
                self.embeddings = HuggingFaceEmbeddings(model_name=config.HUGGINGFACE_MODEL)
            else:
                logger.info("Using OpenAI embeddings")
                if not config.OPENAI_API_KEY:
                    raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env file.")
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=config.OPENAI_API_KEY,
                    model=config.OPENAI_EMBEDDING_MODEL,
                )

            # Initialiser le LLM (uniquement pour OpenAI, HuggingFace nécessiterait une configuration différente)
            if not config.USE_HUGGINGFACE_EMBEDDINGS:
                self.llm = ChatOpenAI(
                    openai_api_key=config.OPENAI_API_KEY,
                    model_name=config.OPENAI_MODEL,
                    temperature=0.7,
                )
            else:
                # Pour les embeddings HuggingFace, vous devez configurer un LLM local ou utiliser une API
                logger.warning("Using HuggingFace embeddings without LLM configured. Q&A will not work.")

        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def ingest_document(self, pdf_bytes: bytes, doc_id: str, doc_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traiter le PDF et créer un magasin vectoriel.

        Args:
            pdf_bytes: Contenu du fichier PDF
            doc_id: Identifiant du document
            doc_metadata: Métadonnées du document (titre, auteur, etc.)

        Returns:
            Dictionnaire avec les statistiques d'ingestion
        """
        try:
            logger.info(f"Starting document ingestion for {doc_id}")

            # Extraire le texte
            raw_text = PDFProcessor.extract_text_from_pdf(pdf_bytes)

            if not raw_text or len(raw_text) < 100:
                raise Exception("PDF appears to be empty or contains too little text")

            # Nettoyer le texte
            cleaned_text = PDFProcessor.clean_text(raw_text)

            # Diviser en chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )

            chunks = text_splitter.split_text(cleaned_text)
            logger.info(f"Split document into {len(chunks)} chunks")

            # Créer des objets Document avec métadonnées
            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        "doc_id": doc_id,
                        "title": doc_metadata.get("title", "Unknown"),
                        "author": doc_metadata.get("author", "Unknown"),
                        "chunk_index": i,
                    },
                )
                for i, chunk in enumerate(chunks)
            ]

            # Créer le magasin vectoriel
            logger.info("Creating vector store...")
            self.vector_store = FAISS.from_documents(documents, self.embeddings)

            # Créer la chaîne de conversation
            if self.llm:
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer",
                )

                self.conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.vector_store.as_retriever(search_kwargs={"k": config.TOP_K_RESULTS}),
                    memory=memory,
                    return_source_documents=True,
                    verbose=config.DEBUG,
                )

            self.current_doc_id = doc_id
            self.chat_history = []

            logger.info("Document ingestion completed successfully")

            return {
                "success": True,
                "doc_id": doc_id,
                "total_chunks": len(chunks),
                "total_characters": len(cleaned_text),
                "chunk_size": config.CHUNK_SIZE,
            }

        except Exception as e:
            logger.error(f"Error during document ingestion: {e}")
            raise Exception(f"Failed to process document: {str(e)}")

    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Poser une question sur le document ingéré.

        Args:
            question: Question de l'utilisateur

        Returns:
            Dictionnaire avec la réponse et les documents sources
        """
        try:
            if not self.vector_store:
                raise Exception("No document has been ingested yet")

            if not self.conversation_chain:
                raise Exception("Conversation chain not initialized. OpenAI API key may be missing.")

            logger.info(f"Processing question: {question}")

            # Obtenir la réponse
            result = self.conversation_chain({"question": question})

            answer = result.get("answer", "")
            source_documents = result.get("source_documents", [])

            # Formater les sources
            sources = []
            for i, doc in enumerate(source_documents):
                sources.append(
                    {
                        "chunk_index": doc.metadata.get("chunk_index", i),
                        "content": doc.page_content[:200] + "...",  # Preview
                        "relevance_score": i + 1,  # Score simulé (FAISS ne retourne pas de scores par défaut)
                    }
                )

            # Stocker dans l'historique de chat
            self.chat_history.append((question, answer))

            logger.info(f"Generated answer with {len(sources)} source documents")

            return {"answer": answer, "sources": sources, "question": question}

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            raise Exception(f"Failed to answer question: {str(e)}")

    def reset(self):
        """Réinitialiser l'état du moteur RAG."""
        logger.info("Resetting RAG engine")
        self.vector_store = None
        self.conversation_chain = None
        self.chat_history = []
        self.current_doc_id = None

    def get_chat_history(self) -> List[Tuple[str, str]]:
        """Obtenir l'historique de conversation."""
        return self.chat_history
