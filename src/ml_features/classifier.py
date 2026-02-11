"""Autoclassification des thèses académiques."""

import logging
from typing import List, Dict, Any
from transformers import pipeline
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThesisClassifier:
    """Classification automatique des thèses par domaine, méthodologie et type."""

    # Mots-clés de domaine pour la classification basée sur des règles
    DOMAIN_KEYWORDS = {
        "Informatique": ["machine learning", "deep learning", "intelligence artificielle",
                         "algorithme", "réseau de neurones", "apprentissage", "ia", "ai",
                         "computer science", "software", "programming", "data science"],
        "Biologie": ["biologie", "génétique", "cellule", "adn", "protéine", "organisme",
                     "écosystème", "évolution", "biochimie", "microbiologie"],
        "Physique": ["physique", "quantique", "particule", "énergie", "matière",
                     "thermodynamique", "mécanique", "électromagnétisme"],
        "Mathématiques": ["mathématiques", "théorème", "preuve", "algèbre", "géométrie",
                         "topologie", "analyse", "probabilité", "statistique"],
        "Médecine": ["médecine", "médical", "patient", "diagnostic", "traitement",
                     "maladie", "santé", "clinique", "thérapie", "pathologie"],
        "Chimie": ["chimie", "molécule", "réaction", "synthèse", "catalyse",
                  "composé", "élément", "solution", "organique"],
        "Sciences Sociales": ["sociologie", "psychologie", "anthropologie", "économie",
                             "politique", "société", "comportement", "social"],
        "Ingénierie": ["ingénierie", "conception", "système", "optimisation",
                      "mécanique", "électrique", "civil", "industriel"],
    }

    METHODOLOGY_KEYWORDS = {
        "Quantitative": ["quantitatif", "statistique", "données", "mesure", "analyse numérique",
                        "expérimentation", "sondage", "questionnaire", "échantillon"],
        "Qualitative": ["qualitatif", "entretien", "observation", "ethnographie",
                       "analyse de contenu", "cas d'étude", "phénoménologie"],
        "Expérimentale": ["expérience", "expérimental", "protocole", "test",
                         "validation", "laboratoire", "essai"],
        "Théorique": ["théorique", "théorie", "modèle", "conceptuel",
                     "abstrait", "formalisation", "axiomatique"],
        "Computationnelle": ["simulation", "modélisation", "computationnel",
                            "numérique", "algorithmique"],
    }

    CONTRIBUTION_TYPES = {
        "Théorique": ["théorie", "modèle théorique", "framework conceptuel",
                     "nouvelle approche théorique"],
        "Appliquée": ["application", "implémentation", "développement",
                     "système", "prototype", "outil"],
        "Revue de littérature": ["revue", "état de l'art", "survey",
                                "littérature", "synthèse"],
        "Méthodologique": ["méthodologie", "méthode nouvelle", "approche méthodologique",
                          "technique", "procédure"],
    }

    def __init__(self, use_ml_model: bool = False):
        """
        Initialiser le classifieur.

        Args:
            use_ml_model: Utiliser ou non le modèle ML (nécessite un GPU) ou basé sur des règles
        """
        self.use_ml_model = use_ml_model
        self.classifier = None

        if use_ml_model:
            try:
                logger.info("Loading ML classification model...")
                # Utiliser la classification zero-shot pour la flexibilité
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )
                logger.info("ML model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load ML model: {e}. Using rule-based classifier.")
                self.use_ml_model = False

    def classify(self, text: str) -> Dict[str, List[str]]:
        """
        Classifier une thèse en fonction de son texte.

        Args:
            text: Résumé de la thèse ou texte complet

        Returns:
            Dictionnaire avec domaines, méthodologies et types de contribution
        """
        text_lower = text.lower()

        # Classification par domaine
        domains = self._classify_domain(text_lower)

        # Classification par méthodologie
        methodologies = self._classify_methodology(text_lower)

        # Classification par type de contribution
        contribution_types = self._classify_contribution(text_lower)

        return {
            "domains": domains,
            "methodologies": methodologies,
            "contribution_types": contribution_types,
        }

    def _classify_domain(self, text: str) -> List[str]:
        """Classifier le domaine scientifique."""
        if self.use_ml_model and self.classifier:
            result = self.classifier(
                text[:512],  # Limite pour le modèle
                list(self.DOMAIN_KEYWORDS.keys()),
                multi_label=True
            )
            # Retourner les étiquettes avec score > 0.3
            return [label for label, score in zip(result['labels'], result['scores'])
                   if score > 0.3][:3]

        # Classification basée sur des règles
        scores = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                scores[domain] = score

        # Retourner les 3 meilleurs domaines
        sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [domain for domain, _ in sorted_domains[:3]]

    def _classify_methodology(self, text: str) -> List[str]:
        """Classifier la méthodologie de recherche."""
        scores = {}
        for methodology, keywords in self.METHODOLOGY_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                scores[methodology] = score

        # Retourner les 2 meilleures méthodologies
        sorted_methods = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [method for method, _ in sorted_methods[:2]]

    def _classify_contribution(self, text: str) -> List[str]:
        """Classifier le type de contribution."""
        scores = {}
        for contrib_type, keywords in self.CONTRIBUTION_TYPES.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                scores[contrib_type] = score

        # Retourner les 2 meilleurs types de contribution
        sorted_contribs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [contrib for contrib, _ in sorted_contribs[:2]]

    def get_filter_options(self, documents: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Obtenir toutes les classifications uniques d'une liste de documents.

        Args:
            documents: Liste de documents classifiés

        Returns:
            Dictionnaire avec tous les domaines, méthodologies, etc. uniques
        """
        all_domains = set()
        all_methodologies = set()
        all_contributions = set()

        for doc in documents:
            if 'classification' in doc:
                all_domains.update(doc['classification'].get('domains', []))
                all_methodologies.update(doc['classification'].get('methodologies', []))
                all_contributions.update(doc['classification'].get('contribution_types', []))

        return {
            "domains": sorted(list(all_domains)),
            "methodologies": sorted(list(all_methodologies)),
            "contribution_types": sorted(list(all_contributions)),
        }
