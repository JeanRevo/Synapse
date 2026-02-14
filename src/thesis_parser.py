"""Parser intelligent pour analyser la structure des thèses académiques."""

import io
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Section:
    """Représente une section de thèse."""
    level: int  # 1=chapitre, 2=section, 3=sous-section
    title: str
    page_start: int
    page_end: Optional[int]
    content: str
    parent_section: Optional[str] = None


class ThesisParser:
    """Parser pour extraire la structure hiérarchique des thèses."""

    # Patterns pour détecter les sections
    CHAPTER_PATTERNS = [
        r'^Chapitre\s+(\d+|[IVX]+)[:\s]+(.+)$',
        r'^Chapter\s+(\d+|[IVX]+)[:\s]+(.+)$',
        r'^CHAPITRE\s+(\d+|[IVX]+)[:\s]+(.+)$',
        r'^(\d+)\.\s+([A-Z][^a-z]{5,})$',  # "1. INTRODUCTION"
        r'^([IVX]+)\.\s+([A-Z][^a-z]{5,})$',  # "I. INTRODUCTION"
    ]

    SECTION_PATTERNS = [
        r'^(\d+)\.(\d+)\s+(.+)$',  # "1.1 Introduction"
        r'^(\d+)\.(\d+)\.\s+(.+)$',  # "1.1. Introduction"
    ]

    SUBSECTION_PATTERNS = [
        r'^(\d+)\.(\d+)\.(\d+)\s+(.+)$',  # "1.1.1 Contexte"
        r'^(\d+)\.(\d+)\.(\d+)\.\s+(.+)$',  # "1.1.1. Contexte"
    ]

    # Sections communes dans les thèses
    COMMON_SECTIONS = [
        "abstract", "résumé", "remerciements", "acknowledgments",
        "introduction", "conclusion", "bibliographie", "bibliography",
        "references", "annexe", "appendix", "table des matières",
        "table of contents", "liste des figures", "list of figures"
    ]

    @staticmethod
    def extract_toc_from_pdf(pdf_bytes: bytes, max_toc_pages: int = 20) -> List[Dict[str, Any]]:
        """
        Extraire la table des matières du PDF.

        Args:
            pdf_bytes: Contenu du PDF
            max_toc_pages: Nombre de pages à analyser pour la TOC

        Returns:
            Liste de sections trouvées dans la TOC
        """
        try:
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
            toc_entries = []

            # Essayer d'utiliser la TOC intégrée du PDF
            pdf_toc = doc.get_toc()
            if pdf_toc and len(pdf_toc) > 0:
                logger.info(f"TOC intégrée trouvée avec {len(pdf_toc)} entrées")
                for entry in pdf_toc:
                    level, title, page = entry
                    toc_entries.append({
                        "level": level,
                        "title": title.strip(),
                        "page": page,
                    })
                doc.close()
                return toc_entries

            # Sinon, analyser manuellement les premières pages
            logger.info("Pas de TOC intégrée, analyse manuelle...")

            for page_num in range(min(max_toc_pages, len(doc))):
                page = doc[page_num]
                text = page.get_text("text")
                lines = text.split('\n')

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Détecter les entrées de TOC (titre + numéro de page)
                    # Format: "1. Introduction ............... 12"
                    # Format: "Chapitre 1: Contexte ......... 25"
                    toc_match = re.search(
                        r'^(.+?)\s*[.…]+\s*(\d+)$',
                        line
                    )

                    if toc_match:
                        title = toc_match.group(1).strip()
                        page = int(toc_match.group(2))

                        # Déterminer le niveau
                        level = ThesisParser._detect_heading_level(title)

                        toc_entries.append({
                            "level": level,
                            "title": title,
                            "page": page,
                        })

            doc.close()
            logger.info(f"Extrait {len(toc_entries)} entrées de TOC")
            return toc_entries

        except Exception as e:
            logger.error(f"Erreur lors de l'extraction de la TOC: {e}")
            return []

    @staticmethod
    def _detect_heading_level(text: str) -> int:
        """
        Détecter le niveau hiérarchique d'un titre.

        Args:
            text: Texte du titre

        Returns:
            Niveau (1=chapitre, 2=section, 3=sous-section)
        """
        text_stripped = text.strip()

        # 1. Mots-clés explicites de chapitre -> niveau 1
        for pattern in ThesisParser.CHAPTER_PATTERNS:
            if re.match(pattern, text_stripped, re.IGNORECASE):
                return 1

        # 2. Sous-section numérotée (X.Y.Z) -> niveau 3
        for pattern in ThesisParser.SUBSECTION_PATTERNS:
            if re.match(pattern, text_stripped):
                return 3

        # 3. Section numérotée (X.Y) -> niveau 2
        for pattern in ThesisParser.SECTION_PATTERNS:
            if re.match(pattern, text_stripped):
                return 2

        # 4. Numérotation simple : "1 Introduction" ou "2 Contexte" -> niveau 1
        if re.match(r'^(\d+)\s+[A-ZÀ-Ú]', text_stripped):
            return 1

        # 5. Numérotation romaine : "I Introduction" -> niveau 1
        if re.match(r'^[IVX]+[\s.:]+\s*\w', text_stripped, re.IGNORECASE):
            return 1

        # 6. Sections communes (Introduction, Conclusion, etc.) -> niveau 1
        text_lower = text_stripped.lower()
        chapter_keywords = [
            "introduction", "conclusion", "bibliographie", "bibliography",
            "references", "résumé", "abstract", "remerciements",
            "acknowledgments", "annexe", "appendix", "table des matières",
            "list of figures", "liste des figures", "glossaire", "glossary",
            "avant-propos", "préface", "préambule",
        ]
        for keyword in chapter_keywords:
            if text_lower.startswith(keyword) or text_lower == keyword:
                return 1

        # 7. Titre tout en MAJUSCULES (souvent un chapitre) -> niveau 1
        words = text_stripped.split()
        if len(words) >= 2 and all(w.isupper() or not w.isalpha() for w in words):
            return 1

        # 8. Par défaut -> niveau 3 (paragraphe, le plus fin)
        return 3

    @staticmethod
    def extract_hierarchical_structure(
        pdf_bytes: bytes,
        use_toc: bool = True
    ) -> List[Section]:
        """
        Extraire la structure hiérarchique complète du PDF.

        Args:
            pdf_bytes: Contenu du PDF
            use_toc: Utiliser la TOC si disponible

        Returns:
            Liste de sections hiérarchiques
        """
        try:
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
            sections = []

            # Essayer d'utiliser la TOC d'abord
            toc_entries = []
            if use_toc:
                toc_entries = ThesisParser.extract_toc_from_pdf(pdf_bytes)

            # Si TOC disponible, l'utiliser pour la structure
            if toc_entries:
                logger.info("Utilisation de la TOC pour structure hiérarchique")

                for i, entry in enumerate(toc_entries):
                    page_start = entry["page"]
                    page_end = toc_entries[i + 1]["page"] - 1 if i + 1 < len(toc_entries) else len(doc)

                    # Extraire le contenu de cette section
                    content = ""
                    for page_num in range(page_start - 1, min(page_end, len(doc))):
                        if page_num < 0 or page_num >= len(doc):
                            continue
                        page = doc[page_num]
                        content += page.get_text("text") + "\n\n"

                    sections.append(Section(
                        level=entry["level"],
                        title=entry["title"],
                        page_start=page_start,
                        page_end=page_end,
                        content=content.strip(),
                    ))

            else:
                # Analyse heuristique sans TOC
                logger.info("Analyse heuristique de la structure (pas de TOC)")

                current_chapter = None
                current_section = None

                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text("text")
                    lines = text.split('\n')

                    for line in lines:
                        line = line.strip()
                        if len(line) < 3 or len(line) > 200:
                            continue

                        # Détecter chapitre
                        for pattern in ThesisParser.CHAPTER_PATTERNS:
                            match = re.match(pattern, line, re.IGNORECASE)
                            if match:
                                # Finaliser le chapitre précédent
                                if current_chapter:
                                    current_chapter.page_end = page_num

                                current_chapter = Section(
                                    level=1,
                                    title=line,
                                    page_start=page_num + 1,
                                    page_end=None,
                                    content="",
                                )
                                sections.append(current_chapter)
                                break

                        # Détecter section
                        for pattern in ThesisParser.SECTION_PATTERNS:
                            match = re.match(pattern, line)
                            if match:
                                if current_section:
                                    current_section.page_end = page_num

                                current_section = Section(
                                    level=2,
                                    title=line,
                                    page_start=page_num + 1,
                                    page_end=None,
                                    content="",
                                    parent_section=current_chapter.title if current_chapter else None,
                                )
                                sections.append(current_section)
                                break

            doc.close()
            logger.info(f"Extrait {len(sections)} sections hiérarchiques")
            return sections

        except Exception as e:
            logger.error(f"Erreur lors de l'extraction de structure: {e}")
            return []

    @staticmethod
    def detect_document_type(pdf_bytes: bytes) -> str:
        """
        Détecter le type de document (thèse, article, rapport).

        Args:
            pdf_bytes: Contenu du PDF

        Returns:
            Type de document détecté
        """
        try:
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")

            # Analyser les 5 premières pages
            first_pages_text = ""
            for page_num in range(min(5, len(doc))):
                page = doc[page_num]
                first_pages_text += page.get_text("text").lower()

            doc.close()

            # Heuristiques de détection
            if any(keyword in first_pages_text for keyword in ["thèse", "thesis", "doctorat", "phd", "doctorate"]):
                return "thesis"
            elif any(keyword in first_pages_text for keyword in ["abstract", "keywords", "journal"]):
                return "article"
            elif "rapport" in first_pages_text or "report" in first_pages_text:
                return "report"
            else:
                return "unknown"

        except Exception as e:
            logger.error(f"Erreur lors de la détection de type: {e}")
            return "unknown"

    @staticmethod
    def extract_metadata(pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Extraire les métadonnées structurelles du PDF.

        Args:
            pdf_bytes: Contenu du PDF

        Returns:
            Métadonnées structurelles
        """
        try:
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")

            # Métadonnées PDF standard
            metadata = doc.metadata or {}

            # Ajouter des infos structurelles
            structural_metadata = {
                "total_pages": len(doc),
                "has_toc": len(doc.get_toc()) > 0 if doc.get_toc() else False,
                "document_type": ThesisParser.detect_document_type(pdf_bytes),
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
            }

            doc.close()
            return structural_metadata

        except Exception as e:
            logger.error(f"Erreur lors de l'extraction de métadonnées: {e}")
            return {}
