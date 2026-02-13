"""Context Expander pour enrichir les chunks avec contexte environnant."""

import logging
from typing import List, Dict, Any, Optional
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextExpander:
    """Expander pour ajouter du contexte aux chunks récupérés."""

    def __init__(self, all_documents: List[Document]):
        """
        Initialiser le Context Expander.

        Args:
            all_documents: Tous les documents du corpus
        """
        self.all_documents = all_documents

        # Créer un index par chunk_index pour accès rapide
        self.doc_index = {}
        for doc in all_documents:
            chunk_idx = doc.metadata.get("chunk_index", -1)
            if chunk_idx >= 0:
                self.doc_index[chunk_idx] = doc

    def expand_context(
        self,
        source_documents: List[Document],
        window_size: int = 1,
        same_page_only: bool = False
    ) -> List[Document]:
        """
        Ajouter du contexte environnant aux chunks récupérés.

        Args:
            source_documents: Documents sources récupérés par le RAG
            window_size: Nombre de chunks avant/après à ajouter
            same_page_only: N'ajouter que chunks de la même page

        Returns:
            Documents avec contexte étendu
        """
        expanded_docs = []
        processed_chunks = set()

        for doc in source_documents:
            chunk_idx = doc.metadata.get("chunk_index", -1)
            page_num = doc.metadata.get("page_num", -1)

            if chunk_idx < 0:
                # Pas d'index, ajouter tel quel
                expanded_docs.append(doc)
                continue

            # Ajouter les chunks environnants
            for offset in range(-window_size, window_size + 1):
                target_idx = chunk_idx + offset

                # Éviter les doublons
                if target_idx in processed_chunks:
                    continue

                # Récupérer le chunk
                target_doc = self.doc_index.get(target_idx)

                if not target_doc:
                    continue

                # Filtrer par page si demandé
                if same_page_only:
                    target_page = target_doc.metadata.get("page_num", -1)
                    if target_page != page_num:
                        continue

                # Marquer comme traité et ajouter
                processed_chunks.add(target_idx)
                expanded_docs.append(target_doc)

        # Trier par chunk_index pour maintenir l'ordre
        expanded_docs.sort(key=lambda d: d.metadata.get("chunk_index", 0))

        logger.info(
            f"Context expanded: {len(source_documents)} → {len(expanded_docs)} chunks "
            f"(window={window_size})"
        )

        return expanded_docs

    def expand_by_chapter(
        self,
        source_documents: List[Document],
    ) -> List[Document]:
        """
        Ajouter tous les chunks du même chapitre.

        Args:
            source_documents: Documents sources

        Returns:
            Documents avec tous les chunks du chapitre
        """
        expanded_docs = []
        processed_chunks = set()
        chapters_added = set()

        for doc in source_documents:
            chapter_title = doc.metadata.get("chapter_title")
            chunk_idx = doc.metadata.get("chunk_index", -1)

            # Ajouter le chunk original
            if chunk_idx >= 0 and chunk_idx not in processed_chunks:
                expanded_docs.append(doc)
                processed_chunks.add(chunk_idx)

            # Si pas de chapitre, continuer
            if not chapter_title or chapter_title in chapters_added:
                continue

            # Marquer le chapitre comme traité
            chapters_added.add(chapter_title)

            # Trouver tous les chunks du même chapitre
            for candidate_doc in self.all_documents:
                candidate_chapter = candidate_doc.metadata.get("chapter_title")
                candidate_idx = candidate_doc.metadata.get("chunk_index", -1)

                if (
                    candidate_chapter == chapter_title
                    and candidate_idx >= 0
                    and candidate_idx not in processed_chunks
                ):
                    expanded_docs.append(candidate_doc)
                    processed_chunks.add(candidate_idx)

        # Trier par chunk_index
        expanded_docs.sort(key=lambda d: d.metadata.get("chunk_index", 0))

        logger.info(
            f"Chapter context: {len(source_documents)} → {len(expanded_docs)} chunks "
            f"({len(chapters_added)} chapters)"
        )

        return expanded_docs

    def expand_by_section(
        self,
        source_documents: List[Document],
    ) -> List[Document]:
        """
        Ajouter tous les chunks de la même section.

        Args:
            source_documents: Documents sources

        Returns:
            Documents avec tous les chunks de la section
        """
        expanded_docs = []
        processed_chunks = set()
        sections_added = set()

        for doc in source_documents:
            section_title = doc.metadata.get("section_title")
            chunk_idx = doc.metadata.get("chunk_index", -1)

            # Ajouter le chunk original
            if chunk_idx >= 0 and chunk_idx not in processed_chunks:
                expanded_docs.append(doc)
                processed_chunks.add(chunk_idx)

            # Si pas de section, continuer
            if not section_title or section_title in sections_added:
                continue

            # Marquer la section comme traitée
            sections_added.add(section_title)

            # Trouver tous les chunks de la même section
            for candidate_doc in self.all_documents:
                candidate_section = candidate_doc.metadata.get("section_title")
                candidate_idx = candidate_doc.metadata.get("chunk_index", -1)

                if (
                    candidate_section == section_title
                    and candidate_idx >= 0
                    and candidate_idx not in processed_chunks
                ):
                    expanded_docs.append(candidate_doc)
                    processed_chunks.add(candidate_idx)

        # Trier par chunk_index
        expanded_docs.sort(key=lambda d: d.metadata.get("chunk_index", 0))

        logger.info(
            f"Section context: {len(source_documents)} → {len(expanded_docs)} chunks "
            f"({len(sections_added)} sections)"
        )

        return expanded_docs

    def merge_overlapping_chunks(
        self,
        source_documents: List[Document]
    ) -> str:
        """
        Fusionner les chunks qui se chevauchent en un seul texte.

        Args:
            source_documents: Documents à fusionner

        Returns:
            Texte fusionné
        """
        if not source_documents:
            return ""

        # Trier par chunk_index
        sorted_docs = sorted(
            source_documents,
            key=lambda d: d.metadata.get("chunk_index", 0)
        )

        # Fusionner les textes
        merged_text = []
        for doc in sorted_docs:
            merged_text.append(doc.page_content)

        return "\n\n".join(merged_text)

    @staticmethod
    def get_page_range(documents: List[Document]) -> tuple:
        """
        Obtenir la plage de pages couverte par les documents.

        Args:
            documents: Liste de documents

        Returns:
            (page_min, page_max)
        """
        if not documents:
            return (0, 0)

        pages = [doc.metadata.get("page_num", 0) for doc in documents]
        return (min(pages), max(pages))

    @staticmethod
    def filter_by_page(
        documents: List[Document],
        page_num: int
    ) -> List[Document]:
        """
        Filtrer les documents par numéro de page.

        Args:
            documents: Liste de documents
            page_num: Numéro de page cible

        Returns:
            Documents de cette page uniquement
        """
        return [
            doc for doc in documents
            if doc.metadata.get("page_num") == page_num
        ]
