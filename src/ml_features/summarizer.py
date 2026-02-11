"""Résumé automatique des thèses académiques."""

import logging
from typing import Dict, Optional
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThesisSummarizer:
    """Résumé automatique multi-niveaux de documents de thèse."""

    def __init__(self, use_transformers: bool = True):
        """
        Initialiser le résumeur.

        Args:
            use_transformers: Utiliser ou non les modèles transformer (nécessite internet)
        """
        self.use_transformers = use_transformers
        self.summarizer = None

        if use_transformers:
            try:
                from transformers import pipeline
                logger.info("Loading summarization model...")
                # Utiliser BART pour un meilleur support du français
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=-1  # Processeur
                )
                logger.info("Summarization model loaded")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}. Using extractive summarization.")
                self.use_transformers = False

    def generate_summaries(self, text: str, title: str = "") -> Dict[str, str]:
        """
        Générer des résumés multi-niveaux.

        Args:
            text: Texte complet à résumer
            title: Titre du document

        Returns:
            Dictionnaire avec résumés tldr, exécutif et détaillé
        """
        # Nettoyer et préparer le texte
        cleaned_text = self._clean_text(text)

        # Générer différents niveaux de résumé
        tldr = self._generate_tldr(cleaned_text, title)
        executive = self._generate_executive(cleaned_text)
        detailed = self._generate_detailed(cleaned_text)

        return {
            "tldr": tldr,
            "executive": executive,
            "detailed": detailed,
        }

    def _clean_text(self, text: str) -> str:
        """Nettoyer le texte pour le résumé."""
        # Supprimer les espaces excessifs
        text = re.sub(r'\s+', ' ', text)

        # Supprimer les URLs
        text = re.sub(r'http\S+|www.\S+', '', text)

        # Supprimer les adresses email
        text = re.sub(r'\S+@\S+', '', text)

        return text.strip()

    def _generate_tldr(self, text: str, title: str) -> str:
        """Générer le TL;DR (3 phrases)."""
        if self.use_transformers and self.summarizer:
            try:
                # Utiliser les 1000 premiers mots pour le TL;DR
                preview = " ".join(text.split()[:1000])
                result = self.summarizer(
                    preview,
                    max_length=100,
                    min_length=50,
                    do_sample=False
                )
                return result[0]['summary_text']
            except Exception as e:
                logger.warning(f"Transformer summarization failed: {e}")

        # Repli: résumé extractif
        return self._extractive_summary(text, num_sentences=3)

    def _generate_executive(self, text: str) -> str:
        """Générer le résumé exécutif (1 paragraphe)."""
        if self.use_transformers and self.summarizer:
            try:
                # Utiliser les 2000 premiers mots
                preview = " ".join(text.split()[:2000])
                result = self.summarizer(
                    preview,
                    max_length=200,
                    min_length=100,
                    do_sample=False
                )
                return result[0]['summary_text']
            except Exception as e:
                logger.warning(f"Transformer summarization failed: {e}")

        # Repli
        return self._extractive_summary(text, num_sentences=6)

    def _generate_detailed(self, text: str) -> str:
        """Generate detailed summary (1 page)."""
        if self.use_transformers and self.summarizer:
            try:
                # Use first 5000 words
                preview = " ".join(text.split()[:5000])
                result = self.summarizer(
                    preview,
                    max_length=500,
                    min_length=300,
                    do_sample=False
                )
                return result[0]['summary_text']
            except Exception as e:
                logger.warning(f"Transformer summarization failed: {e}")

        # Repli
        return self._extractive_summary(text, num_sentences=15)

    def _extractive_summary(self, text: str, num_sentences: int = 3) -> str:
        """
        Simple extractive summarization (fallback).

        Selects most important sentences based on:
        - Position (first sentences are important)
        - Keyword frequency
        - Sentence length
        """
        # Diviser en phrases
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return "No summary available."

        # Noter les phrases
        scored_sentences = []
        for i, sentence in enumerate(sentences[:50]):  # Considérer les 50 premières phrases
            score = 0

            # Score de position (plus tôt est mieux)
            score += (50 - i) * 0.1

            # Score de longueur (préférer longueur moyenne)
            word_count = len(sentence.split())
            if 10 < word_count < 30:
                score += 2

            # Score des mots-clés
            important_words = ['résultat', 'conclusion', 'propose', 'démontre',
                             'analyse', 'étude', 'recherche', 'méthode']
            score += sum(1 for word in important_words if word in sentence.lower())

            scored_sentences.append((score, sentence))

        # Trier par score et prendre le top N
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [s for _, s in scored_sentences[:num_sentences]]

        return " ".join(top_sentences) + "."

    def summarize_sections(self, text: str) -> Dict[str, str]:
        """
        Identify and summarize major sections.

        Args:
            text: Full text

        Returns:
            Dictionary mapping section names to summaries
        """
        sections = self._identify_sections(text)
        summaries = {}

        for section_name, section_text in sections.items():
            if len(section_text.split()) > 50:
                summary = self._extractive_summary(section_text, num_sentences=2)
                summaries[section_name] = summary

        return summaries

    def _identify_sections(self, text: str) -> Dict[str, str]:
        """Identify major sections in academic text."""
        # Common section headers in French academic papers
        section_patterns = [
            r'(?:^|\n)\s*(?:I+\.|Chapitre \d+|INTRODUCTION)\s*:?\s*([^\n]+)',
            r'(?:^|\n)\s*(?:II+\.|Chapitre \d+|MÉTHODOLOGIE|MÉTHODE)\s*:?\s*([^\n]+)',
            r'(?:^|\n)\s*(?:III+\.|Chapitre \d+|RÉSULTATS)\s*:?\s*([^\n]+)',
            r'(?:^|\n)\s*(?:IV+\.|Chapitre \d+|DISCUSSION)\s*:?\s*([^\n]+)',
            r'(?:^|\n)\s*(?:V+\.|Chapitre \d+|CONCLUSION)\s*:?\s*([^\n]+)',
        ]

        sections = {}
        lines = text.split('\n')

        current_section = "Introduction"
        current_content = []

        for line in lines:
            # Check if this is a section header
            is_header = False
            for pattern in section_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    # Save previous section
                    if current_content:
                        sections[current_section] = "\n".join(current_content)

                    # Start new section
                    current_section = line.strip()
                    current_content = []
                    is_header = True
                    break

            if not is_header:
                current_content.append(line)

        # Save last section
        if current_content:
            sections[current_section] = "\n".join(current_content)

        return sections
