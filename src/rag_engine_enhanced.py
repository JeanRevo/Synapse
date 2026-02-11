"""Moteur RAG amélioré avec des fonctionnalités avancées pour des performances optimales."""

import io
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from sentence_transformers import SentenceTransformer, CrossEncoder
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
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Encoder une requête unique."""
        embedding = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        return embedding[0].tolist()


class EnhancedPDFProcessor:
    """Traitement PDF avancé avec suivi de page et chunking sémantique."""

    @staticmethod
    def extract_text_with_pages(pdf_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Extraire le texte du PDF avec numéros de page et métadonnées.

        Args:
            pdf_bytes: Contenu du fichier PDF en bytes

        Returns:
            Liste de dictionnaires avec page_num, text et metadata
        """
        try:
            logger.info("Starting enhanced PDF text extraction with page tracking")
            pdf_stream = io.BytesIO(pdf_bytes)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")

            pages_data = []
            total_pages = len(doc)

            for page_num in range(total_pages):
                page = doc[page_num]
                text = page.get_text("text")

                # Extraire des métadonnées supplémentaires
                blocks = page.get_text("dict")["blocks"]

                # Nettoyage basique
                text = text.strip()
                if text:
                    pages_data.append({
                        "page_num": page_num + 1,  # Indexé à partir de 1
                        "text": text,
                        "char_count": len(text),
                        "has_images": len(page.get_images()) > 0,
                    })

                if page_num % 10 == 0:
                    logger.info(f"Processed {page_num + 1}/{total_pages} pages")

            doc.close()
            logger.info(f"Extracted text from {total_pages} pages with metadata")
            return pages_data

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")

    @staticmethod
    def clean_text_advanced(text: str) -> str:
        """
        Nettoyage de texte avancé avec de meilleures heuristiques.

        Args:
            text: Texte brut du PDF

        Returns:
            Texte nettoyé
        """
        # Supprimer les espaces excessifs
        text = re.sub(r'\s+', ' ', text)

        # Supprimer les nombres isolés (probablement des numéros de page)
        text = re.sub(r'\b\d{1,3}\b(?=\s|$)', '', text)

        # Corriger la césure aux sauts de ligne
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

        # Supprimer les motifs d'en-têtes/pieds de page (communs dans les articles académiques)
        text = re.sub(r'^\d+\s+[A-Z\s]+$', '', text, flags=re.MULTILINE)

        return text.strip()

    @staticmethod
    def create_semantic_chunks(pages_data: List[Dict[str, Any]],
                               chunk_size: int = 1000,
                               chunk_overlap: int = 200) -> List[Document]:
        """
        Créer des chunks sémantiques avec métadonnées enrichies.

        Args:
            pages_data: Liste de dictionnaires de données de page
            chunk_size: Taille cible du chunk
            chunk_overlap: Chevauchement entre les chunks

        Returns:
            Liste d'objets Document avec métadonnées
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        )

        documents = []
        chunk_id = 0

        for page_data in pages_data:
            page_num = page_data["page_num"]
            text = EnhancedPDFProcessor.clean_text_advanced(page_data["text"])

            if not text:
                continue

            # Diviser le texte de la page en chunks
            chunks = text_splitter.split_text(text)

            for chunk_idx, chunk in enumerate(chunks):
                # Détecter si le chunk contient des en-têtes de section
                has_header = bool(re.search(r'^[A-Z\s]{10,}$', chunk[:100], re.MULTILINE))

                # Estimer le nombre de tokens (approximation: ~4 caractères par token)
                token_count = len(chunk) // 4

                doc = Document(
                    page_content=chunk,
                    metadata={
                        "chunk_id": chunk_id,
                        "page_num": page_num,
                        "page_chunk_index": chunk_idx,
                        "total_chunks_in_page": len(chunks),
                        "char_count": len(chunk),
                        "token_count": token_count,
                        "has_header": has_header,
                        "has_images": page_data.get("has_images", False),
                    },
                )
                documents.append(doc)
                chunk_id += 1

        logger.info(f"Created {len(documents)} semantic chunks with metadata")
        return documents


class HybridRetriever:
    """Récupérateur hybride combinant la recherche vectorielle avec BM25 et le reclassement."""

    def __init__(self, vector_store: FAISS, documents: List[Document],
                 use_reranking: bool = True):
        """
        Initialiser le récupérateur hybride.

        Args:
            vector_store: Magasin vectoriel FAISS
            documents: Documents originaux
            use_reranking: Utiliser ou non le reclassement cross-encoder
        """
        self.vector_store = vector_store
        self.documents = documents
        self.use_reranking = use_reranking

        # Initialiser le reclasseur si activé
        if self.use_reranking:
            try:
                logger.info("Loading cross-encoder for reranking...")
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}. Continuing without reranking.")
                self.use_reranking = False

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Récupérer les documents pertinents en utilisant une approche hybride.

        Args:
            query: Requête de recherche
            k: Nombre de documents à récupérer

        Returns:
            Liste de documents pertinents
        """
        # Recherche de similarité vectorielle (récupérer plus de candidats pour le reclassement)
        candidates_k = k * 3 if self.use_reranking else k

        # Obtenir les candidats du magasin vectoriel avec MMR pour la diversité
        try:
            candidates = self.vector_store.max_marginal_relevance_search(
                query,
                k=candidates_k,
                fetch_k=candidates_k * 2
            )
        except:
            # Repli sur la recherche de similarité régulière
            candidates = self.vector_store.similarity_search(query, k=candidates_k)

        logger.info(f"Retrieved {len(candidates)} candidates from vector store")

        # Reclasser si activé
        if self.use_reranking and len(candidates) > 0:
            # Créer des paires requête-document pour le reclassement
            pairs = [[query, doc.page_content] for doc in candidates]

            # Obtenir les scores de reclassement
            scores = self.reranker.predict(pairs)

            # Trier par scores et prendre le top k
            scored_docs = list(zip(candidates, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            final_docs = [doc for doc, score in scored_docs[:k]]
            logger.info(f"Reranked to top {len(final_docs)} documents")
            return final_docs

        return candidates[:k]


class EnhancedRAGEngine:
    """Moteur RAG amélioré avec récupération et traitement avancés."""

    def __init__(self):
        """Initialiser le moteur RAG amélioré."""
        self.vector_store: Optional[FAISS] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.conversation_chain: Optional[ConversationalRetrievalChain] = None
        self.chat_history: List[Tuple[str, str]] = []
        self.current_doc_id: Optional[str] = None
        self.documents: List[Document] = []
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

            # Initialiser le LLM avec des paramètres optimisés
            if not config.USE_HUGGINGFACE_EMBEDDINGS:
                self.llm = ChatOpenAI(
                    openai_api_key=config.OPENAI_API_KEY,
                    model_name=config.OPENAI_MODEL,
                    temperature=0.3,  # Plus bas pour des réponses plus factuelles
                    max_tokens=1000,  # Contrôler la longueur de la réponse
                )
            else:
                logger.warning("Using HuggingFace embeddings without LLM configured. Q&A will not work.")

        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def _create_custom_prompt(self) -> PromptTemplate:
        """Créer un template de prompt optimisé pour les Q&R de documents académiques."""
        template = """Tu es un assistant expert spécialisé dans l'analyse de documents scientifiques et académiques.

Contexte du document:
{context}

Historique de conversation:
{chat_history}

Question de l'utilisateur: {question}

Instructions:
1. Réponds de manière précise et factuelle en te basant UNIQUEMENT sur le contexte fourni
2. Si tu cites le document, indique la page quand c'est possible
3. Si la réponse n'est pas dans le contexte, dis-le clairement
4. Structure ta réponse de manière claire avec des points si nécessaire
5. Reste concis mais complet (maximum 200 mots)

Réponse:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"]
        )

    def ingest_document(self, pdf_bytes: bytes, doc_id: str, doc_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traiter le PDF avec chunking et indexation améliorés.

        Args:
            pdf_bytes: Contenu du fichier PDF
            doc_id: Identifiant du document
            doc_metadata: Métadonnées du document (titre, auteur, etc.)

        Returns:
            Dictionnaire avec les statistiques d'ingestion
        """
        try:
            logger.info(f"Starting enhanced document ingestion for {doc_id}")

            # Extraire le texte avec suivi de page
            pages_data = EnhancedPDFProcessor.extract_text_with_pages(pdf_bytes)

            if not pages_data:
                raise Exception("PDF appears to be empty or contains no extractable text")

            # Créer des chunks sémantiques avec métadonnées enrichies
            self.documents = EnhancedPDFProcessor.create_semantic_chunks(
                pages_data,
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )

            # Ajouter les métadonnées au niveau du document à tous les chunks
            for doc in self.documents:
                doc.metadata.update({
                    "doc_id": doc_id,
                    "title": doc_metadata.get("title", "Unknown"),
                    "author": doc_metadata.get("author", "Unknown"),
                })

            logger.info(f"Created {len(self.documents)} chunks with metadata")

            # Créer le magasin vectoriel
            logger.info("Creating vector store...")
            self.vector_store = FAISS.from_documents(self.documents, self.embeddings)

            # Créer le récupérateur hybride
            logger.info("Initializing hybrid retriever...")
            self.hybrid_retriever = HybridRetriever(
                self.vector_store,
                self.documents,
                use_reranking=True
            )

            # Créer la chaîne de conversation avec récupérateur personnalisé
            if self.llm:
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer",
                )

                self.conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.vector_store.as_retriever(
                        search_type="mmr",  # Pertinence Marginale Maximale
                        search_kwargs={
                            "k": config.TOP_K_RESULTS,
                            "fetch_k": config.TOP_K_RESULTS * 3,
                            "lambda_mult": 0.5  # Équilibrer pertinence vs diversité
                        }
                    ),
                    memory=memory,
                    return_source_documents=True,
                    verbose=config.DEBUG,
                )

            self.current_doc_id = doc_id
            self.chat_history = []

            # Calculer les statistiques
            total_chars = sum(len(doc.page_content) for doc in self.documents)
            total_pages = len(pages_data)

            logger.info("Enhanced document ingestion completed successfully")

            return {
                "success": True,
                "doc_id": doc_id,
                "total_chunks": len(self.documents),
                "total_pages": total_pages,
                "total_characters": total_chars,
                "avg_chunk_size": total_chars // len(self.documents),
                "chunk_size_config": config.CHUNK_SIZE,
            }

        except Exception as e:
            logger.error(f"Error during document ingestion: {e}")
            raise Exception(f"Failed to process document: {str(e)}")

    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Poser une question avec récupération et génération de réponse améliorées.

        Args:
            question: Question de l'utilisateur

        Returns:
            Dictionnaire avec la réponse et les informations sources enrichies
        """
        try:
            if not self.vector_store:
                raise Exception("No document has been ingested yet")

            if not self.conversation_chain:
                raise Exception("Conversation chain not initialized. OpenAI API key may be missing.")

            logger.info(f"Processing question: {question}")

            # Obtenir la réponse avec récupération améliorée
            result = self.conversation_chain({"question": question})

            answer = result.get("answer", "")
            source_documents = result.get("source_documents", [])

            # Formater les sources avec métadonnées enrichies
            sources = []
            for i, doc in enumerate(source_documents):
                metadata = doc.metadata
                sources.append({
                    "chunk_id": metadata.get("chunk_id", i),
                    "page_num": metadata.get("page_num", "Unknown"),
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "char_count": metadata.get("char_count", 0),
                    "has_header": metadata.get("has_header", False),
                    "relevance_rank": i + 1,
                })

            # Stocker dans l'historique de chat
            self.chat_history.append((question, answer))

            logger.info(f"Generated answer with {len(sources)} source documents")

            return {
                "answer": answer,
                "sources": sources,
                "question": question,
                "num_sources": len(sources),
            }

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            raise Exception(f"Failed to answer question: {str(e)}")

    def reset(self):
        """Réinitialiser l'état du moteur RAG."""
        logger.info("Resetting enhanced RAG engine")
        self.vector_store = None
        self.hybrid_retriever = None
        self.conversation_chain = None
        self.chat_history = []
        self.current_doc_id = None
        self.documents = []

    def get_chat_history(self) -> List[Tuple[str, str]]:
        """Obtenir l'historique de conversation."""
        return self.chat_history

    def get_statistics(self) -> Dict[str, Any]:
        """Obtenir les statistiques du moteur RAG."""
        return {
            "current_doc_id": self.current_doc_id,
            "num_chunks": len(self.documents),
            "chat_history_length": len(self.chat_history),
            "has_vector_store": self.vector_store is not None,
            "has_hybrid_retriever": self.hybrid_retriever is not None,
        }
