"""Moteur RAG pour le traitement PDF et les réponses aux questions."""

import io
import logging
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from .config import config
from .thesis_parser import ThesisParser
from .hierarchical_chunker import HierarchicalChunker, HierarchicalChunk
from .multi_index_rag import MultiIndexRAG
from .query_router import QueryRouter
from .context_expander import ContextExpander

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
    def extract_text_with_pages(pdf_bytes: bytes) -> List[Tuple[int, str]]:
        """
        Extraire le texte des bytes PDF avec numéros de page.

        Args:
            pdf_bytes: Contenu du fichier PDF en bytes

        Returns:
            Liste de tuples (numéro_page, texte) avec page 1-indexed
        """
        try:
            logger.info("Starting PDF text extraction with page tracking")
            pdf_stream = io.BytesIO(pdf_bytes)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")

            pages_text = []
            total_pages = len(doc)

            for page_num in range(total_pages):
                page = doc[page_num]
                text = page.get_text("text").strip()

                if text:
                    pages_text.append((page_num + 1, text))  # 1-indexed

                if page_num % 10 == 0:
                    logger.info(f"Processed {page_num + 1}/{total_pages} pages")

            doc.close()
            logger.info(f"Extracted text from {len(pages_text)} pages")
            return pages_text

        except Exception as e:
            logger.error(f"Error extracting text with pages from PDF: {e}")
            raise Exception(f"Failed to extract text with pages from PDF: {str(e)}")

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
        self.documents: List[Document] = []  # Stocker les documents pour BM25
        self.hybrid_retriever = None  # Retriever hybride (FAISS + BM25)

        # Phase 2: Hierarchical + Multi-Index
        self.multi_index_rag: Optional[MultiIndexRAG] = None
        self.hierarchical_chunker: Optional[HierarchicalChunker] = None
        self.use_multi_index = config.USE_MULTI_INDEX

        # Phase 3: Query Router + Context Expander
        self.query_router: Optional[QueryRouter] = None
        self.context_expander: Optional[ContextExpander] = None

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

            # Extraire le texte avec numéros de page
            pages_text = PDFProcessor.extract_text_with_pages(pdf_bytes)

            if not pages_text or len(pages_text) == 0:
                raise Exception("PDF appears to be empty or contains no text")

            documents = []
            total_chars = sum(len(text) for _, text in pages_text)
            ingestion_stats = {
                "chunking_method": "flat",
                "total_pages": len(pages_text),
            }

            # Phase 2: Essayer le chunking hiérarchique
            if config.USE_HIERARCHICAL_CHUNKING:
                logger.info("🔍 Tentative de chunking hiérarchique...")

                try:
                    # Extraire la structure de la thèse
                    sections = ThesisParser.extract_hierarchical_structure(pdf_bytes, use_toc=True)

                    if sections and len(sections) > 0:
                        logger.info(f"✅ Structure détectée: {len(sections)} sections")

                        # Créer le chunker hiérarchique
                        self.hierarchical_chunker = HierarchicalChunker(
                            chapter_chunk_size=config.CHAPTER_CHUNK_SIZE,
                            section_chunk_size=config.SECTION_CHUNK_SIZE,
                            paragraph_chunk_size=config.PARAGRAPH_CHUNK_SIZE,
                            chunk_overlap=config.CHUNK_OVERLAP,
                        )

                        # Enrichir métadonnées
                        enriched_metadata = {
                            **doc_metadata,
                            "doc_id": doc_id,
                        }

                        # Créer chunks hiérarchiques
                        hierarchical_chunks = self.hierarchical_chunker.create_hierarchical_chunks(
                            sections,
                            enriched_metadata
                        )

                        # Convertir en Documents LangChain
                        documents = self.hierarchical_chunker.convert_to_documents(hierarchical_chunks)

                        # IMPORTANT: Ajouter les pages non couvertes par la structure
                        # (titre, remerciements, abstract, bibliographie, etc.)
                        all_page_nums = {page_num for page_num, _ in pages_text}
                        covered_pages = set()
                        for section in sections:
                            if section.page_start and section.page_end:
                                for p in range(section.page_start, section.page_end + 1):
                                    covered_pages.add(p)
                            elif section.page_start:
                                covered_pages.add(section.page_start)

                        uncovered_pages = [
                            (page_num, text) for page_num, text in pages_text
                            if page_num not in covered_pages
                        ]

                        logger.info(f"📊 Pages totales: {len(all_page_nums)}, couvertes: {len(covered_pages)}, non couvertes: {len(uncovered_pages)}")
                        if uncovered_pages:
                            logger.info(f"📄 Pages non couvertes: {[p for p, _ in uncovered_pages[:20]]}")
                            logger.info(f"📄 {len(uncovered_pages)} pages non couvertes par la structure, ajout en chunks plats...")
                            flat_docs = self._create_flat_chunks(uncovered_pages, doc_id, doc_metadata)
                            documents.extend(flat_docs)

                            # Mettre à jour les index des chunks ajoutés
                            max_idx = max(d.metadata.get("chunk_index", 0) for d in documents) if documents else 0
                            for i, doc in enumerate(flat_docs):
                                doc.metadata["chunk_index"] = max_idx + 1 + i

                            # Recréer les hierarchical_chunks avec les flat docs
                            for doc in flat_docs:
                                hierarchical_chunks.append(
                                    HierarchicalChunk(
                                        content=doc.page_content,
                                        level=3,
                                        page_num=doc.metadata["page_num"],
                                        chunk_index=doc.metadata["chunk_index"],
                                        metadata=doc.metadata,
                                    )
                                )

                        ingestion_stats["chunking_method"] = "hierarchical"
                        ingestion_stats["sections_found"] = len(sections)
                        ingestion_stats["uncovered_pages"] = len(uncovered_pages)

                        # Grouper par niveau pour stats
                        grouped = self.hierarchical_chunker.group_chunks_by_level(hierarchical_chunks)
                        ingestion_stats["chapter_chunks"] = len(grouped.get(1, []))
                        ingestion_stats["section_chunks"] = len(grouped.get(2, []))
                        ingestion_stats["paragraph_chunks"] = len(grouped.get(3, []))

                    else:
                        logger.warning("❌ Pas de structure détectée, fallback sur chunking plat")
                        documents = self._create_flat_chunks(pages_text, doc_id, doc_metadata)

                except Exception as e:
                    logger.warning(f"❌ Échec chunking hiérarchique: {e}, fallback sur chunking plat")
                    documents = self._create_flat_chunks(pages_text, doc_id, doc_metadata)

            else:
                # Chunking plat classique
                logger.info("📄 Chunking plat (hiérarchique désactivé)")
                documents = self._create_flat_chunks(pages_text, doc_id, doc_metadata)

            logger.info(f"Split document into {len(documents)} chunks across {len(pages_text)} pages")

            # Diagnostic: vérifier si des chunks contiennent des infos clés
            uni_chunks = [d for d in documents if "universit" in d.page_content.lower()]
            logger.info(f"🔍 Diagnostic: {len(uni_chunks)} chunks contiennent 'universit'")
            for i, d in enumerate(uni_chunks[:3]):
                logger.info(f"  - Chunk page {d.metadata.get('page_num')}, level {d.metadata.get('level')}: {d.page_content[:150]}...")

            # Stocker les documents pour BM25
            self.documents = documents

            # Phase 2: Créer Multi-Index si activé
            if config.USE_MULTI_INDEX and ingestion_stats["chunking_method"] == "hierarchical":
                logger.info("🔢 Création du Multi-Index (4 indexes spécialisés)...")

                self.multi_index_rag = MultiIndexRAG(self.embeddings)
                self.multi_index_rag.build_indexes(hierarchical_chunks, documents)

                # Utiliser le retriever full par défaut
                self.hybrid_retriever = self.multi_index_rag.get_retriever("full")

                ingestion_stats["multi_index_enabled"] = True
                ingestion_stats["index_stats"] = self.multi_index_rag.get_index_stats()

            else:
                # Créer le magasin vectoriel FAISS simple (par batch)
                logger.info("Creating FAISS vector store...")
                self.vector_store = self._create_faiss_batched(documents, self.embeddings)

                # Créer le retriever hybride (Phase 1: MMR + BM25)
                # (Seulement quand multi-index n'est PAS utilisé, car il a ses propres retrievers)
                if config.USE_HYBRID_SEARCH:
                    logger.info("Creating hybrid retriever (MMR + BM25)...")

                    faiss_retriever = self.vector_store.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": config.TOP_K_RESULTS,
                            "fetch_k": config.TOP_K_RESULTS * config.MMR_FETCH_K_MULTIPLIER,
                            "lambda_mult": config.MMR_LAMBDA,
                        }
                    )

                    bm25_retriever = BM25Retriever.from_documents(documents)
                    bm25_retriever.k = config.TOP_K_RESULTS

                    self.hybrid_retriever = EnsembleRetriever(
                        retrievers=[faiss_retriever, bm25_retriever],
                        weights=[config.FAISS_WEIGHT, config.BM25_WEIGHT],
                    )

                    logger.info(f"✅ Hybrid RAG: MMR ({config.FAISS_WEIGHT*100}%) + BM25 ({config.BM25_WEIGHT*100}%)")
                else:
                    self.hybrid_retriever = self.vector_store.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": config.TOP_K_RESULTS,
                            "fetch_k": config.TOP_K_RESULTS * config.MMR_FETCH_K_MULTIPLIER,
                            "lambda_mult": config.MMR_LAMBDA,
                        }
                    )

            # Phase 3: Initialiser Query Router et Context Expander
            logger.info("🧭 Initialisation Query Router et Context Expander...")

            # Query Router
            self.query_router = QueryRouter(llm=self.llm)

            # Context Expander
            self.context_expander = ContextExpander(all_documents=documents)

            logger.info("✅ Phase 3 initialisée (Query Router + Context Expander)")

            # Créer la chaîne de conversation avec le retriever hybride
            if self.llm:
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer",
                )

                self.conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.hybrid_retriever,  # Utiliser le retriever hybride
                    memory=memory,
                    return_source_documents=True,
                    verbose=config.DEBUG,
                )

                logger.info("✅ Hybrid RAG initialized: MMR (60%) + BM25 (40%)")

            self.current_doc_id = doc_id
            self.chat_history = []

            logger.info("Document ingestion completed successfully")

            # Retour avec stats enrichies
            return {
                "success": True,
                "doc_id": doc_id,
                "total_chunks": len(documents),
                "total_characters": total_chars,
                "chunk_size": config.CHUNK_SIZE,
                **ingestion_stats,  # Inclure stats de chunking et multi-index
            }

        except Exception as e:
            logger.error(f"Error during document ingestion: {e}")
            raise Exception(f"Failed to process document: {str(e)}")

    @staticmethod
    def _create_faiss_batched(documents: List[Document], embeddings, batch_size: int = 100) -> FAISS:
        """
        Créer un FAISS vector store par batch pour éviter la limite de tokens OpenAI.

        Args:
            documents: Documents à indexer
            embeddings: Modèle d'embeddings
            batch_size: Nombre de documents par batch

        Returns:
            FAISS vector store
        """
        if len(documents) <= batch_size:
            return FAISS.from_documents(documents, embeddings)

        logger.info(f"Création FAISS par batch ({len(documents)} docs, batch_size={batch_size})...")

        # Premier batch pour créer l'index
        vector_store = FAISS.from_documents(documents[:batch_size], embeddings)

        # Batches suivants
        for i in range(batch_size, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            logger.info(f"  Batch {i // batch_size + 1}: {len(batch)} documents...")
            vector_store.add_documents(batch)

        logger.info(f"✅ FAISS créé: {len(documents)} documents indexés")
        return vector_store

    def _create_flat_chunks(
        self,
        pages_text: List[Tuple[int, str]],
        doc_id: str,
        doc_metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Créer des chunks plats (méthode classique).

        Args:
            pages_text: Liste de (page_num, text)
            doc_id: ID du document
            doc_metadata: Métadonnées

        Returns:
            Liste de Documents LangChain
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        documents = []
        chunk_index = 0

        for page_num, page_text in pages_text:
            # Nettoyer le texte de la page
            cleaned_page_text = PDFProcessor.clean_text(page_text)

            # Diviser le texte de la page en chunks
            page_chunks = text_splitter.split_text(cleaned_page_text)

            for chunk in page_chunks:
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "doc_id": doc_id,
                            "title": doc_metadata.get("title", "Unknown"),
                            "author": doc_metadata.get("author", "Unknown"),
                            "chunk_index": chunk_index,
                            "page_num": page_num,
                            "level": 3,  # Par défaut niveau paragraphe
                            "chunk_type": "paragraph",
                        },
                    )
                )
                chunk_index += 1

        return documents

    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Poser une question sur le document ingéré.

        Args:
            question: Question de l'utilisateur

        Returns:
            Dictionnaire avec la réponse et les documents sources
        """
        try:
            logger.info(f"Processing question: {question}")

            # Phase 3: Utiliser le Multi-Index avec Query Router si disponible
            if self.multi_index_rag and self.query_router:
                return self._ask_question_advanced(question)

            # Fallback: méthode classique avec ConversationalRetrievalChain
            if not self.vector_store:
                raise Exception("No document has been ingested yet")

            if not self.conversation_chain:
                raise Exception("Conversation chain not initialized. OpenAI API key may be missing.")

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
                        "page_num": doc.metadata.get("page_num", 1),
                        "content": doc.page_content[:200] + "...",
                        "full_content": doc.page_content,
                        "relevance_score": i + 1,
                    }
                )

            # Stocker dans l'historique de chat
            self.chat_history.append((question, answer))

            logger.info(f"Generated answer with {len(sources)} source documents")

            return {"answer": answer, "sources": sources, "question": question}

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            raise Exception(f"Failed to answer question: {str(e)}")

    def _filter_relevant_sources(
        self,
        answer: str,
        source_documents: List[Document],
        question: str,
        max_sources: int = 5
    ) -> List[Document]:
        """
        Filtrer les sources pour ne garder que celles pertinentes à la réponse.

        Règle stricte : une source DOIT contenir au moins un mot-clé de la question.
        Ensuite, scoring par mots-clés de la réponse pour le classement.

        Args:
            answer: Réponse générée par le LLM
            source_documents: Documents sources candidats
            question: Question originale
            max_sources: Nombre max de sources à retourner

        Returns:
            Sources filtrées par pertinence
        """
        stop_words = {
            "le", "la", "les", "de", "du", "des", "un", "une", "et", "ou",
            "pour", "dans", "est", "ce", "cette", "qui", "que", "quoi",
            "par", "sur", "avec", "sans", "son", "sa", "ses", "au", "aux",
            "en", "à", "il", "elle", "ils", "elles", "pas", "ne", "se",
            "on", "nous", "vous", "leur", "leurs", "tout", "tous", "très",
            "plus", "aussi", "mais", "car", "donc", "comme", "bien", "peut",
            "fait", "été", "sont", "ont", "ces", "d'un", "d'une", "cette",
            "c'est", "n'est", "l'on", "qu'il", "qu'elle", "qu'un", "qu'une",
            "quel", "quelle", "quels", "quelles",
        }

        # Mots-clés de la question (critère obligatoire)
        question_words = set(question.lower().split())
        question_keywords = {w.strip(".,;:!?()\"'") for w in question_words if len(w) > 3} - stop_words

        # Mots-clés de la réponse (critère de scoring)
        answer_words = set(answer.lower().split())
        answer_keywords = {w.strip(".,;:!?()\"'") for w in answer_words if len(w) > 3} - stop_words

        if not question_keywords and not answer_keywords:
            return source_documents[:max_sources]

        scored_sources = []
        for doc in source_documents:
            content_lower = doc.page_content.lower()

            # FILTRE STRICT : la source DOIT contenir au moins un mot-clé de la question
            if question_keywords:
                has_question_match = any(kw in content_lower for kw in question_keywords)
                if not has_question_match:
                    continue  # Éliminer cette source

            # Scoring par mots-clés de la réponse
            score = sum(1 for kw in answer_keywords if kw in content_lower)
            scored_sources.append((score, doc))

        scored_sources.sort(key=lambda x: x[0], reverse=True)

        result = [doc for _, doc in scored_sources[:max_sources]]

        logger.info(f"🎯 Filtre pertinence: {len(result)}/{len(source_documents)} sources (question_kw: {question_keywords})")
        for i, (s, d) in enumerate(scored_sources[:max_sources]):
            logger.info(f"  - Source {i}: score={s}, page {d.metadata.get('page_num')}, {d.page_content[:80]}...")

        return result

    def _keyword_search_documents(self, query: str, max_results: int = 5) -> List[Document]:
        """
        Recherche par mots-clés dans tous les documents (fallback pour métadonnées).

        Args:
            query: Question de l'utilisateur
            max_results: Nombre max de résultats

        Returns:
            Documents triés par pertinence mot-clé
        """
        # Mots vides français à ignorer
        stop_words = {
            "le", "la", "les", "de", "du", "des", "un", "une", "et", "ou",
            "pour", "dans", "est", "ce", "cette", "quel", "quelle", "quels",
            "quelles", "qui", "que", "quoi", "comment", "où", "par", "sur",
            "avec", "sans", "son", "sa", "ses", "au", "aux", "en", "à",
        }

        query_words = set(query.lower().split())
        keywords = query_words - stop_words

        if not keywords:
            return []

        scored_docs = []
        for doc in self.documents:
            content_lower = doc.page_content.lower()
            # Score = nombre de mots-clés trouvés dans le contenu
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > 0:
                scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        results = [doc for _, doc in scored_docs[:max_results]]
        if results:
            logger.info(f"🔑 Keyword search: {len(results)} documents trouvés pour mots-clés {keywords}")
            for i, doc in enumerate(results[:3]):
                logger.info(f"  - Doc {i}: page {doc.metadata.get('page_num')}, {doc.page_content[:100]}...")
        return results

    def _ask_question_advanced(self, question: str) -> Dict[str, Any]:
        """
        Poser une question avec Query Router + Multi-Index + Context Expander.

        Args:
            question: Question de l'utilisateur

        Returns:
            Dictionnaire avec la réponse et les documents sources
        """
        try:
            # 1. Router la question vers l'index approprié
            index_type = self.query_router.route_query(question)
            logger.info(f"📍 Question routée vers index: {index_type.upper()}")

            # 2. Récupérer le retriever approprié
            retriever = self.multi_index_rag.get_retriever(index_type)

            # 3. Récupérer les documents
            source_documents = retriever.get_relevant_documents(question)

            logger.info(f"Retrieved {len(source_documents)} documents from {index_type} index")

            # 3b. Pour les questions métadonnées/full, compléter avec recherche par mots-clés
            if index_type == "full":
                keyword_docs = self._keyword_search_documents(question)
                existing_contents = {doc.page_content[:100] for doc in source_documents}
                added = 0
                for doc in keyword_docs:
                    if doc.page_content[:100] not in existing_contents:
                        source_documents.insert(0, doc)  # Priorité aux résultats mot-clé
                        existing_contents.add(doc.page_content[:100])
                        added += 1
                if added > 0:
                    logger.info(f"🔑 Ajouté {added} documents par recherche mot-clé")

            # 4. Expansion du contexte si nécessaire
            if self.context_expander and len(source_documents) > 0:
                # Expansion légère (window=1) pour questions précises
                if index_type == "paragraph":
                    source_documents = self.context_expander.expand_context(
                        source_documents,
                        window_size=1,
                        same_page_only=True
                    )
                # Expansion par section pour questions de section
                elif index_type == "section":
                    source_documents = self.context_expander.expand_by_section(source_documents)

            # 5. Construire le contexte pour le LLM
            context = "\n\n---\n\n".join([doc.page_content for doc in source_documents[:config.TOP_K_RESULTS]])

            # 6. Construire le prompt avec historique
            history_context = ""
            if self.chat_history:
                recent_history = self.chat_history[-3:]  # 3 dernières QA
                history_context = "\n\nHistorique récent:\n"
                for q, a in recent_history:
                    history_context += f"Q: {q}\nR: {a}\n\n"

            prompt = f"""Tu es un assistant spécialisé dans l'analyse de thèses académiques.
Réponds à la question en te basant sur le contexte fourni. Sois précis et complet.
Si l'information n'est pas explicitement dans le contexte, essaie de déduire la réponse à partir des éléments disponibles (mentions d'université, laboratoire, auteur, etc.).
{history_context}
Contexte:
{context}

Question: {question}

Réponse:"""

            # 7. Générer la réponse avec le LLM
            if not self.llm:
                raise Exception("LLM not initialized")

            answer = self.llm.predict(prompt)

            # 7b. Filtrer les sources pour ne garder que celles pertinentes à la réponse
            relevant_docs = self._filter_relevant_sources(
                answer, source_documents[:config.TOP_K_RESULTS], question
            )

            # 8. Formater les sources (seulement les pertinentes)
            sources = []
            for i, doc in enumerate(relevant_docs):
                sources.append(
                    {
                        "chunk_index": doc.metadata.get("chunk_index", i),
                        "page_num": doc.metadata.get("page_num", 1),
                        "content": doc.page_content[:200] + "...",
                        "full_content": doc.page_content,
                        "relevance_score": i + 1,
                        "chunk_type": doc.metadata.get("chunk_type", "unknown"),
                        "chapter_title": doc.metadata.get("chapter_title"),
                        "section_title": doc.metadata.get("section_title"),
                    }
                )

            # 9. Stocker dans l'historique
            self.chat_history.append((question, answer))

            logger.info(
                f"✅ Answer generated with {len(sources)} relevant sources "
                f"(index: {index_type}, retrieved: {len(source_documents)}, filtered: {len(relevant_docs)})"
            )

            return {
                "answer": answer,
                "sources": sources,
                "question": question,
                "index_used": index_type,
                "total_context_chunks": len(source_documents),
            }

        except Exception as e:
            logger.error(f"Error in advanced question answering: {e}")
            raise Exception(f"Failed to answer question: {str(e)}")

    def reset(self):
        """Réinitialiser l'état du moteur RAG."""
        logger.info("Resetting RAG engine")
        self.vector_store = None
        self.conversation_chain = None
        self.chat_history = []
        self.current_doc_id = None
        self.documents = []
        self.hybrid_retriever = None

        # Phase 2
        if self.multi_index_rag:
            self.multi_index_rag.reset()
        self.multi_index_rag = None
        self.hierarchical_chunker = None

        # Phase 3
        self.query_router = None
        self.context_expander = None

    def get_chat_history(self) -> List[Tuple[str, str]]:
        """Obtenir l'historique de conversation."""
        return self.chat_history
