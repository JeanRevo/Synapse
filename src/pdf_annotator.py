"""Module d'annotation PDF pour le surlignage et la navigation."""

import io
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFAnnotator:
    """Gérer les annotations et la navigation dans les PDF."""

    # Couleurs pour les différentes sources
    COLORS = ["#FFFF00", "#00FFFF", "#90EE90", "#FFB6C1", "#FFA500", "#DDA0DD", "#87CEEB", "#F0E68C"]

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normaliser le texte pour améliorer la correspondance."""
        # Remplacer les sauts de ligne multiples par un espace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def _search_text_in_page(page, text: str, search_neighbors: bool = False) -> list:
        """
        Rechercher du texte dans une page PDF avec plusieurs stratégies de fallback.

        Args:
            page: Page PyMuPDF
            text: Texte à rechercher
            search_neighbors: Non utilisé (réservé)

        Returns:
            Liste de quads trouvés
        """
        if not text or len(text.strip()) < 5:
            return []

        # Normaliser le texte
        normalized = PDFAnnotator._normalize_text(text)

        # Stratégie 1: Premiers 150 chars
        search_text = normalized[:150]
        quads = page.search_for(search_text, quads=True)
        if quads:
            return quads

        # Stratégie 2: Premiers 80 chars
        search_text = normalized[:80]
        quads = page.search_for(search_text, quads=True)
        if quads:
            return quads

        # Stratégie 3: Premiers 40 chars
        search_text = normalized[:40]
        quads = page.search_for(search_text, quads=True)
        if quads:
            return quads

        # Stratégie 4: Première phrase complète
        sentences = re.split(r'[.!?]\s', normalized)
        for sentence in sentences[:3]:
            sentence = sentence.strip()
            if len(sentence) > 15:
                quads = page.search_for(sentence[:120], quads=True)
                if quads:
                    return quads

        # Stratégie 5: Lignes individuelles du texte original
        lines = text.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if len(line) > 15:
                quads = page.search_for(line[:100], quads=True)
                if quads:
                    return quads

        # Stratégie 6: Mots significatifs (groupes de 3-5 mots consécutifs)
        words = normalized.split()
        if len(words) >= 5:
            # Essayer des séquences de 5 mots
            for start in range(0, min(len(words) - 4, 10)):
                phrase = ' '.join(words[start:start + 5])
                if len(phrase) > 15:
                    quads = page.search_for(phrase, quads=True)
                    if quads:
                        return quads

        return []

    @staticmethod
    def find_text_coordinates(pdf_bytes: bytes, text: str, page_num: int) -> List[Dict[str, float]]:
        """
        Trouver les coordonnées d'un texte dans le PDF.

        Args:
            pdf_bytes: Contenu du PDF en bytes
            text: Texte à rechercher
            page_num: Numéro de page (1-indexed)

        Returns:
            Liste de bounding boxes [{x0, y0, x1, y1}, ...]
        """
        try:
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")

            if page_num < 1 or page_num > len(doc):
                logger.warning(f"Page {page_num} hors limites (total: {len(doc)})")
                doc.close()
                return []

            page = doc[page_num - 1]  # 0-indexed
            quads = PDFAnnotator._search_text_in_page(page, text)

            # Si pas trouvé, essayer les pages voisines (page_num peut être approximatif)
            if not quads:
                for offset in [1, -1, 2, -2]:
                    neighbor_idx = page_num - 1 + offset
                    if 0 <= neighbor_idx < len(doc):
                        quads = PDFAnnotator._search_text_in_page(doc[neighbor_idx], text)
                        if quads:
                            logger.info(f"Texte trouvé sur page voisine {neighbor_idx + 1} au lieu de {page_num}")
                            break

            # Convertir quads en bounding boxes
            boxes = []
            for quad in quads:
                boxes.append({
                    "x0": quad.ul.x,
                    "y0": quad.ul.y,
                    "x1": quad.lr.x,
                    "y1": quad.lr.y,
                    "width": quad.lr.x - quad.ul.x,
                    "height": quad.lr.y - quad.ul.y,
                })

            doc.close()

            if not boxes:
                logger.warning(f"Texte non trouvé sur page {page_num}: {text[:50]}...")

            return boxes

        except Exception as e:
            logger.error(f"Erreur lors de la recherche de coordonnées: {e}")
            return []

    @staticmethod
    def create_annotation_dict(page_num: int, box: Dict[str, float], color: str = "#FFFF00") -> Dict[str, Any]:
        """
        Créer un dictionnaire d'annotation pour streamlit-pdf-viewer.

        Args:
            page_num: Numéro de page (1-indexed)
            box: Bounding box {x0, y0, x1, y1, width, height}
            color: Couleur de surlignage (hex)

        Returns:
            Dictionnaire d'annotation
        """
        return {
            "page": page_num,
            "x": box["x0"],
            "y": box["y0"],
            "width": box["width"],
            "height": box["height"],
            "color": color,
        }

    @staticmethod
    def generate_annotations_from_sources(
        pdf_bytes: bytes,
        sources: List[Dict[str, Any]],
        max_sources: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Générer des annotations pour streamlit-pdf-viewer depuis les sources RAG.

        Args:
            pdf_bytes: Contenu du PDF
            sources: Liste des sources retournées par le RAG
            max_sources: Nombre maximum de sources à annoter

        Returns:
            Liste d'annotations pour streamlit-pdf-viewer
        """
        annotations = []

        for i, source in enumerate(sources[:max_sources]):
            page_num = source.get("page_num", 1)
            content = source.get("content", "")
            full_content = source.get("full_content", content)

            color = PDFAnnotator.COLORS[i % len(PDFAnnotator.COLORS)]
            boxes = PDFAnnotator.find_text_coordinates(pdf_bytes, full_content, page_num)

            for box in boxes:
                annotation = PDFAnnotator.create_annotation_dict(page_num, box, color)
                annotations.append(annotation)

        logger.info(f"Généré {len(annotations)} annotations pour {len(sources[:max_sources])} sources")
        return annotations

    @staticmethod
    def create_navigation_link(page_num: int, source_index: int) -> str:
        """
        Créer un lien de navigation HTML vers une page spécifique.

        Args:
            page_num: Numéro de page
            source_index: Index de la source

        Returns:
            HTML du lien
        """
        color = PDFAnnotator.COLORS[source_index % len(PDFAnnotator.COLORS)]
        return f"""
        <div style="
            display: inline-block;
            padding: 2px 8px;
            margin: 2px;
            background-color: {color};
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: bold;
            cursor: pointer;
        ">
            📄 Page {page_num}
        </div>
        """

    @staticmethod
    def get_page_dimensions(pdf_bytes: bytes, page_num: int) -> Optional[Tuple[float, float]]:
        """
        Obtenir les dimensions d'une page PDF.

        Args:
            pdf_bytes: Contenu du PDF
            page_num: Numéro de page (1-indexed)

        Returns:
            Tuple (width, height) ou None
        """
        try:
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")

            if page_num < 1 or page_num > len(doc):
                doc.close()
                return None

            page = doc[page_num - 1]
            rect = page.rect

            doc.close()
            return (rect.width, rect.height)

        except Exception as e:
            logger.error(f"Erreur lors de la récupération des dimensions: {e}")
            return None

    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
        """
        Convertir une couleur hex en tuple RGB (valeurs 0-1 pour PyMuPDF).

        Args:
            hex_color: Couleur en format hex (#RRGGBB)

        Returns:
            Tuple (r, g, b) avec valeurs entre 0 et 1
        """
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        return (r, g, b)

    @staticmethod
    def _extract_answer_key_phrases(answer_text: str) -> List[str]:
        """
        Extraire les termes-clés de la réponse pour le surlignage.

        Détecte les noms propres, institutions, dates, etc.

        Args:
            answer_text: Texte de la réponse du LLM

        Returns:
            Liste de phrases-clés à surligner
        """
        phrases = []

        # Pattern 1: Noms propres composés (Université de Rennes, École Polytechnique, etc.)
        # Mot majuscule + (de/du/des/la/le/les/l'/d') + Mot majuscule
        proper_noun_pattern = r"[A-ZÀ-Ú][a-zà-ú]+(?:\s+(?:de|du|des|la|le|les|l'|d')\s*)*[A-ZÀ-Ú][a-zà-ú]+"
        matches = re.findall(proper_noun_pattern, answer_text)
        for m in matches:
            m = m.strip()
            if len(m) > 5:
                phrases.append(m)

        # Pattern 2: Mots tout en majuscules (UNIVERSITÉ DE RENNES, CNRS, etc.)
        caps_pattern = r"[A-ZÀ-Ú]{2,}(?:\s+(?:DE|DU|DES|LA|LE|LES)\s+[A-ZÀ-Ú]{2,})*"
        matches = re.findall(caps_pattern, answer_text)
        for m in matches:
            m = m.strip()
            if len(m) > 3:
                phrases.append(m)

        # Pattern 3: Années (2020-2029)
        year_pattern = r"\b(20[0-9]{2})\b"
        matches = re.findall(year_pattern, answer_text)
        phrases.extend(matches)

        # Dédupliquer et trier par longueur (plus long d'abord)
        seen = set()
        unique = []
        for p in phrases:
            p_lower = p.lower()
            if p_lower not in seen and len(p) > 3:
                seen.add(p_lower)
                unique.append(p)

        unique.sort(key=len, reverse=True)

        logger.info(f"🔤 Phrases-clés extraites de la réponse: {unique[:10]}")
        return unique[:10]  # Max 10 phrases

    @staticmethod
    def generate_highlighted_pdf(
        pdf_bytes: bytes,
        sources: List[Dict[str, Any]],
        max_sources: int = 8,
        answer_text: str = None
    ) -> bytes:
        """
        Générer un PDF avec surlignages des sources ET des termes de la réponse.

        Deux types de surlignage:
        1. Surlignage des sources (chunks) - couleurs variées par source
        2. Surlignage des termes de la réponse - rouge/orange sur TOUTES les pages

        Args:
            pdf_bytes: Contenu du PDF original
            sources: Liste des sources retournées par le RAG
            max_sources: Nombre maximum de sources à surligner
            answer_text: Texte de la réponse (pour surligner les termes-clés)

        Returns:
            PDF annoté en bytes
        """
        try:
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")

            annotations_count = 0
            sources_highlighted = 0

            # === PARTIE 1: Surlignage des termes de la réponse (prioritaire) ===
            answer_highlights = 0
            if answer_text:
                key_phrases = PDFAnnotator._extract_answer_key_phrases(answer_text)
                # Couleur rouge/orange pour les termes de la réponse
                answer_color = (1.0, 0.6, 0.0)  # Orange vif

                for phrase in key_phrases:
                    # Chercher sur TOUTES les pages du document
                    for page_idx in range(len(doc)):
                        page = doc[page_idx]
                        # Chercher le texte tel quel
                        quads = page.search_for(phrase, quads=True)
                        # Aussi chercher en minuscules/majuscules variantes
                        if not quads:
                            quads = page.search_for(phrase.upper(), quads=True)
                        if not quads:
                            quads = page.search_for(phrase.title(), quads=True)

                        for quad in quads:
                            try:
                                highlight = page.add_highlight_annot(quad)
                                highlight.set_colors(stroke=answer_color)
                                highlight.set_opacity(0.5)
                                highlight.update()
                                answer_highlights += 1
                            except Exception:
                                pass

                if answer_highlights > 0:
                    logger.info(f"🔤 {answer_highlights} surlignages de termes de la réponse")

            # === PARTIE 2: Surlignage des sources (chunks) ===
            for i, source in enumerate(sources[:max_sources]):
                page_num = source.get("page_num", 1)
                full_content = source.get("full_content", source.get("content", ""))

                if not full_content or page_num < 1 or page_num > len(doc):
                    continue

                color_hex = PDFAnnotator.COLORS[i % len(PDFAnnotator.COLORS)]
                color_rgb = PDFAnnotator.hex_to_rgb(color_hex)

                page = doc[page_num - 1]  # 0-indexed

                # Utiliser la recherche multi-stratégie
                quads = PDFAnnotator._search_text_in_page(page, full_content)

                # Si pas trouvé, essayer les pages voisines
                actual_page = page
                if not quads:
                    for offset in [1, -1, 2, -2]:
                        neighbor_idx = page_num - 1 + offset
                        if 0 <= neighbor_idx < len(doc):
                            quads = PDFAnnotator._search_text_in_page(doc[neighbor_idx], full_content)
                            if quads:
                                actual_page = doc[neighbor_idx]
                                logger.info(f"Source {i}: texte trouvé sur page voisine {neighbor_idx + 1}")
                                break

                if quads:
                    for quad in quads:
                        try:
                            highlight = actual_page.add_highlight_annot(quad)
                            highlight.set_colors(stroke=color_rgb)
                            highlight.set_opacity(0.3)
                            highlight.update()
                            annotations_count += 1
                        except Exception as e:
                            logger.warning(f"Impossible de surligner sur page {page_num}: {e}")
                    sources_highlighted += 1
                else:
                    logger.warning(f"Source {i} (page {page_num}): aucun texte trouvé pour surlignage - '{full_content[:60]}...'")

            total = annotations_count + answer_highlights
            logger.info(f"PDF annoté: {total} surlignages total ({answer_highlights} réponse + {annotations_count} sources, {sources_highlighted}/{len(sources[:max_sources])} sources trouvées)")

            pdf_bytes_annotated = doc.tobytes()
            doc.close()

            return pdf_bytes_annotated

        except Exception as e:
            logger.error(f"Erreur lors de la génération du PDF annoté: {e}")
            return pdf_bytes
