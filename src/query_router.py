"""Query Router intelligent pour diriger les questions vers le bon index."""

import re
import logging
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from .config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryRouter:
    """Router intelligent pour diriger les questions vers l'index approprié."""

    # Patterns pour détecter le type de question
    CHAPTER_PATTERNS = [
        r'chapitre\s+(\d+|[ivx]+)',
        r'chapter\s+(\d+|[ivx]+)',
        r'vue\s+d\'ensemble',
        r'résumé\s+général',
        r'contexte\s+général',
        r'introduction',
        r'conclusion',
    ]

    SECTION_PATTERNS = [
        r'section\s+(\d+\.?\d*)',
        r'partie\s+(\d+\.?\d*)',
        r'méthod(e|ologie)',
        r'résultats',
        r'discussion',
        r'expérience',
    ]

    PAGE_PATTERNS = [
        r'page\s+(\d+)',
        r'p\.\s*(\d+)',
        r'à\s+la\s+page\s+(\d+)',
    ]

    PRECISE_PATTERNS = [
        r'précisément',
        r'exactement',
        r'spécifiquement',
        r'détail',
        r'citation',
        r'définition\s+de',
        r'qu\'est-ce\s+que',
        r'comment\s+fonctionne',
    ]

    # Questions sur les métadonnées du document (titre, auteur, université, date, etc.)
    METADATA_PATTERNS = [
        r'universit[ée]',
        r'auteur',
        r'qui\s+a\s+[ée]crit',
        r'directeur',
        r'encadrant',
        r'laboratoire',
        r'date',
        r'ann[ée]e',
        r'titre',
        r'de\s+quoi\s+parle',
        r'résumé',
        r'sujet',
        r'thème',
        r'domaine',
        r'école\s+doctorale',
    ]

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialiser le Query Router.

        Args:
            llm: LLM pour classification intelligente (optionnel)
        """
        self.llm = llm

    def route_query(self, query: str) -> str:
        """
        Router la question vers l'index approprié.

        Args:
            query: Question de l'utilisateur

        Returns:
            Type d'index: "chapter", "section", "paragraph", ou "full"
        """
        query_lower = query.lower()

        # 1. Détection de page spécifique → paragraph
        if self._match_patterns(query_lower, self.PAGE_PATTERNS):
            logger.info("🎯 Route: PARAGRAPH (page spécifique)")
            return "paragraph"

        # 2. Détection de métadonnées (université, auteur, etc.) → full
        if self._match_patterns(query_lower, self.METADATA_PATTERNS):
            logger.info("🎯 Route: FULL (question métadonnées)")
            return "full"

        # 3. Détection de question précise → paragraph
        if self._match_patterns(query_lower, self.PRECISE_PATTERNS):
            logger.info("🎯 Route: PARAGRAPH (question précise)")
            return "paragraph"

        # 4. Détection de chapitre → chapter
        if self._match_patterns(query_lower, self.CHAPTER_PATTERNS):
            logger.info("🎯 Route: CHAPTER (contexte large)")
            return "chapter"

        # 5. Détection de section → section
        if self._match_patterns(query_lower, self.SECTION_PATTERNS):
            logger.info("🎯 Route: SECTION (contexte moyen)")
            return "section"

        # 6. Par défaut → full (le plus sûr, cherche partout)
        logger.info("🎯 Route: FULL (défaut)")
        return "full"

    def route_query_with_llm(self, query: str) -> str:
        """
        Router la question en utilisant le LLM pour classification.

        Args:
            query: Question de l'utilisateur

        Returns:
            Type d'index recommandé
        """
        if not self.llm:
            # Fallback sur règles heuristiques
            return self.route_query(query)

        try:
            # Prompt pour le LLM
            classification_prompt = f"""Analyze this question and classify it into ONE of these categories:

CATEGORIES:
- "chapter": Questions about general context, overview, introduction, or conclusion
- "section": Questions about specific methodology, results, or discussion sections
- "paragraph": Questions requiring precise details, definitions, or specific pages
- "full": General questions requiring information from multiple parts

QUESTION: {query}

Answer with ONLY one word: chapter, section, paragraph, or full."""

            response = self.llm.predict(classification_prompt)
            route = response.strip().lower()

            if route in ["chapter", "section", "paragraph", "full"]:
                logger.info(f"🤖 LLM Route: {route.upper()}")
                return route
            else:
                logger.warning(f"LLM returned invalid route: {route}, using heuristic")
                return self.route_query(query)

        except Exception as e:
            logger.error(f"Error in LLM routing: {e}, falling back to heuristic")
            return self.route_query(query)

    def _match_patterns(self, text: str, patterns: list) -> bool:
        """
        Vérifier si le texte matche un des patterns.

        Args:
            text: Texte à analyser
            patterns: Liste de regex patterns

        Returns:
            True si au moins un pattern matche
        """
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def get_routing_explanation(self, query: str, route: str) -> str:
        """
        Expliquer pourquoi cette route a été choisie.

        Args:
            query: Question
            route: Route choisie

        Returns:
            Explication lisible
        """
        explanations = {
            "chapter": "Question générale nécessitant un contexte large (chapitre entier)",
            "section": "Question sur une section spécifique (méthodologie, résultats)",
            "paragraph": "Question précise nécessitant des détails exacts",
            "full": "Question générale nécessitant recherche dans tout le document",
        }

        return explanations.get(route, "Route par défaut")

    @staticmethod
    def extract_page_number(query: str) -> Optional[int]:
        """
        Extraire le numéro de page d'une question.

        Args:
            query: Question de l'utilisateur

        Returns:
            Numéro de page ou None
        """
        page_match = re.search(r'page\s+(\d+)|p\.\s*(\d+)', query, re.IGNORECASE)
        if page_match:
            page_num = page_match.group(1) or page_match.group(2)
            return int(page_num)
        return None

    @staticmethod
    def extract_chapter_number(query: str) -> Optional[str]:
        """
        Extraire le numéro de chapitre d'une question.

        Args:
            query: Question de l'utilisateur

        Returns:
            Numéro de chapitre ou None
        """
        chapter_match = re.search(
            r'chapitre\s+(\d+|[ivx]+)|chapter\s+(\d+|[ivx]+)',
            query,
            re.IGNORECASE
        )
        if chapter_match:
            chapter_num = chapter_match.group(1) or chapter_match.group(2)
            return chapter_num
        return None
