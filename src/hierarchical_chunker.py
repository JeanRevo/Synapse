"""Chunking hiérarchique pour thèses académiques (3 niveaux)."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .thesis_parser import ThesisParser, Section

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HierarchicalChunk:
    """Chunk hiérarchique avec contexte."""
    content: str
    level: int  # 1=chapitre, 2=section, 3=paragraphe
    page_num: int
    chunk_index: int
    metadata: Dict[str, Any]

    # Contexte hiérarchique
    chapter_title: Optional[str] = None
    section_title: Optional[str] = None
    parent_chunk_index: Optional[int] = None


class HierarchicalChunker:
    """Chunker hiérarchique à 3 niveaux (chapitre/section/paragraphe)."""

    def __init__(
        self,
        chapter_chunk_size: int = 1500,
        section_chunk_size: int = 1000,
        paragraph_chunk_size: int = 500,
        chunk_overlap: int = 200,
    ):
        """
        Initialiser le chunker hiérarchique.

        Args:
            chapter_chunk_size: Taille max pour chunks de chapitre
            section_chunk_size: Taille max pour chunks de section
            paragraph_chunk_size: Taille max pour chunks de paragraphe
            chunk_overlap: Overlap entre chunks
        """
        self.chapter_chunk_size = chapter_chunk_size
        self.section_chunk_size = section_chunk_size
        self.paragraph_chunk_size = paragraph_chunk_size
        self.chunk_overlap = chunk_overlap

        # Text splitters pour chaque niveau
        self.chapter_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chapter_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n\n", "\n\n", "\n", ". ", " "],
        )

        self.section_splitter = RecursiveCharacterTextSplitter(
            chunk_size=section_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

        self.paragraph_splitter = RecursiveCharacterTextSplitter(
            chunk_size=paragraph_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

    def create_hierarchical_chunks(
        self,
        sections: List[Section],
        doc_metadata: Dict[str, Any]
    ) -> List[HierarchicalChunk]:
        """
        Créer des chunks hiérarchiques à partir de sections.

        Args:
            sections: Sections extraites par ThesisParser
            doc_metadata: Métadonnées du document

        Returns:
            Liste de chunks hiérarchiques
        """
        all_chunks = []
        chunk_index = 0

        current_chapter = None
        current_section = None

        for section in sections:
            # Niveau 1: Chapitre
            if section.level == 1:
                current_chapter = section.title
                current_section = None

                # Créer chunks de chapitre (grands chunks de contexte)
                chapter_texts = self.chapter_splitter.split_text(section.content)

                for i, text in enumerate(chapter_texts):
                    chunk = HierarchicalChunk(
                        content=text,
                        level=1,
                        page_num=section.page_start,
                        chunk_index=chunk_index,
                        metadata={
                            **doc_metadata,
                            "chunk_type": "chapter",
                            "chapter_title": section.title,
                            "section_title": None,
                            "chunk_size": len(text),
                        },
                        chapter_title=section.title,
                        section_title=None,
                    )
                    all_chunks.append(chunk)
                    chunk_index += 1

            # Niveau 2: Section
            elif section.level == 2:
                current_section = section.title

                # Créer chunks de section (taille moyenne)
                section_texts = self.section_splitter.split_text(section.content)

                for i, text in enumerate(section_texts):
                    chunk = HierarchicalChunk(
                        content=text,
                        level=2,
                        page_num=section.page_start,
                        chunk_index=chunk_index,
                        metadata={
                            **doc_metadata,
                            "chunk_type": "section",
                            "chapter_title": current_chapter,
                            "section_title": section.title,
                            "chunk_size": len(text),
                        },
                        chapter_title=current_chapter,
                        section_title=section.title,
                    )
                    all_chunks.append(chunk)
                    chunk_index += 1

            # Niveau 3: Sous-section ou contenu général
            else:
                # Créer chunks de paragraphe (petits chunks précis)
                paragraph_texts = self.paragraph_splitter.split_text(section.content)

                for i, text in enumerate(paragraph_texts):
                    chunk = HierarchicalChunk(
                        content=text,
                        level=3,
                        page_num=section.page_start,
                        chunk_index=chunk_index,
                        metadata={
                            **doc_metadata,
                            "chunk_type": "paragraph",
                            "chapter_title": current_chapter,
                            "section_title": current_section,
                            "chunk_size": len(text),
                        },
                        chapter_title=current_chapter,
                        section_title=current_section,
                    )
                    all_chunks.append(chunk)
                    chunk_index += 1

        logger.info(f"Créé {len(all_chunks)} chunks hiérarchiques")
        return all_chunks

    def create_flat_chunks_with_hierarchy(
        self,
        pages_text: List[tuple],
        doc_metadata: Dict[str, Any]
    ) -> List[HierarchicalChunk]:
        """
        Créer des chunks plats avec métadonnées hiérarchiques enrichies.

        Cette méthode est un fallback quand la structure n'est pas détectable.

        Args:
            pages_text: Liste de (page_num, text)
            doc_metadata: Métadonnées du document

        Returns:
            Liste de chunks hiérarchiques (niveau paragraphe par défaut)
        """
        all_chunks = []
        chunk_index = 0

        for page_num, page_text in pages_text:
            # Utiliser le splitter de paragraphe par défaut
            texts = self.paragraph_splitter.split_text(page_text)

            for text in texts:
                chunk = HierarchicalChunk(
                    content=text,
                    level=3,  # Niveau paragraphe par défaut
                    page_num=page_num,
                    chunk_index=chunk_index,
                    metadata={
                        **doc_metadata,
                        "chunk_type": "paragraph",
                        "chapter_title": None,
                        "section_title": None,
                        "chunk_size": len(text),
                    },
                    chapter_title=None,
                    section_title=None,
                )
                all_chunks.append(chunk)
                chunk_index += 1

        logger.info(f"Créé {len(all_chunks)} chunks plats avec métadonnées")
        return all_chunks

    def convert_to_documents(
        self,
        hierarchical_chunks: List[HierarchicalChunk]
    ) -> List[Document]:
        """
        Convertir chunks hiérarchiques en Documents LangChain.

        Args:
            hierarchical_chunks: Chunks hiérarchiques

        Returns:
            Liste de Documents LangChain
        """
        documents = []

        for chunk in hierarchical_chunks:
            doc = Document(
                page_content=chunk.content,
                metadata={
                    **chunk.metadata,
                    "chunk_index": chunk.chunk_index,
                    "page_num": chunk.page_num,
                    "level": chunk.level,
                    "chapter_title": chunk.chapter_title,
                    "section_title": chunk.section_title,
                }
            )
            documents.append(doc)

        return documents

    @staticmethod
    def group_chunks_by_level(
        chunks: List[HierarchicalChunk]
    ) -> Dict[int, List[HierarchicalChunk]]:
        """
        Grouper les chunks par niveau hiérarchique.

        Args:
            chunks: Liste de chunks hiérarchiques

        Returns:
            Dictionnaire {level: [chunks]}
        """
        grouped = {1: [], 2: [], 3: []}

        for chunk in chunks:
            if chunk.level in grouped:
                grouped[chunk.level].append(chunk)

        logger.info(
            f"Chunks groupés: "
            f"{len(grouped[1])} chapitres, "
            f"{len(grouped[2])} sections, "
            f"{len(grouped[3])} paragraphes"
        )

        return grouped

    @staticmethod
    def get_chunk_context(
        chunk: HierarchicalChunk,
        all_chunks: List[HierarchicalChunk],
        context_window: int = 1
    ) -> str:
        """
        Obtenir le contexte d'un chunk (chunks environnants).

        Args:
            chunk: Chunk cible
            all_chunks: Tous les chunks
            context_window: Nombre de chunks avant/après

        Returns:
            Contexte enrichi
        """
        chunk_idx = chunk.chunk_index

        # Trouver les chunks environnants
        start_idx = max(0, chunk_idx - context_window)
        end_idx = min(len(all_chunks), chunk_idx + context_window + 1)

        context_chunks = all_chunks[start_idx:end_idx]
        context_text = "\n\n".join([c.content for c in context_chunks])

        return context_text
