"""Support d'internationalisation (i18n) pour Synapse."""

from typing import Dict, Any


class Translations:
    """Gérer les traductions pour plusieurs langues."""

    LANGUAGES = {
        "en": "English",
        "fr": "Français",
    }

    TEXTS = {
        "en": {
            # Header
            "app_title": "🧠 Synapse",
            "app_subtitle": "Your AI assistant for exploring scientific theses",

            # Sidebar
            "sidebar_brand": "🧠 Synapse",
            "sidebar_brand_tagline": "AI Scientific Assistant",
            "sidebar_status_ready": "Ready to explore",
            "sidebar_status_analyzing": "Analyzing document",
            "sidebar_doc_loaded": "Document loaded:",
            "sidebar_new_search": "🔄 New Search",
            "sidebar_history_title": "Recent conversations",
            "sidebar_history_loading": "Restoring conversation...",
            "sidebar_powered_by": "Powered by",
            "sidebar_version": "v1.0",

            # Phase 1: Search
            "search_header": "🔍 Explore HAL Archives",
            "search_input_label": "Enter your research topic or keywords:",
            "search_input_placeholder": "e.g., machine learning, climate change, quantum computing...",
            "search_input_help": "Search for theses in the HAL Science database",
            "search_num_results": "Number of results",
            "search_button": "Search",
            "search_searching": "🔎 Searching HAL archives...",
            "search_no_results": "No results found. Try different keywords.",
            "search_results_found": "Found {total} theses! Showing {shown} results.",
            "search_error": "Error during search: {error}",
            "search_results_header": "📚 Search Results ({total} total)",

            # Document card
            "doc_author": "**Author:**",
            "doc_date": "**Date:**",
            "doc_domain": "**Domain:**",
            "doc_keywords": "**Keywords:**",
            "doc_abstract": "**Abstract:**",
            "doc_view_hal": "🔗 View on HAL",
            "doc_chat_button": "💬 Chat with this document",
            "doc_pdf_unavailable": "PDF not available",

            # Phase 2: Chat
            "chat_header": "💬 Chat with Document",
            "chat_current_doc": "📄 Current Document",
            "chat_doc_title": "**Title:**",
            "chat_doc_author": "**Author:**",
            "chat_doc_date": "**Date:**",
            "chat_doc_view_hal": "🔗 View on HAL",
            "chat_input_placeholder": "Ask a question about the document...",
            "chat_thinking": "🤔 Thinking...",
            "chat_sources_header": "📚 View Sources",
            "chat_source_chunk": "Source {num} (Chunk {chunk})",
            "chat_error": "Error: {error}",
            "chat_show_pdf": "📄 Show PDF",
            "chat_hide_pdf": "📄 Hide PDF",
            "chat_pdf_viewer": "📄 PDF Document",

            # Document loading
            "loading_title": "📥 Loading document...",
            "loading_downloading": "Downloading PDF...",
            "loading_initializing": "Initializing RAG engine...",
            "loading_processing": "Processing document (this may take a minute)...",
            "loading_success": "✅ Document loaded! Processed {chunks} chunks ({chars:,} characters)",
            "loading_error": "Error loading document: {error}",

            # ML Features
            "ml_filters_header": "🏷️ Smart Filters",
            "ml_domains": "Domains",
            "ml_methodologies": "Methodologies",
            "ml_generate_summary": "📝 Generate Summary",
            "ml_generating": "Generating summary...",
            "ml_tldr": "**TL;DR**",
            "ml_executive_summary": "**Executive Summary**",
            "ml_detailed_summary": "**Detailed Summary**",
            "ml_view_full": "View full summary",
            "ml_similar_theses": "🔗 Similar Theses",
            "ml_similarity": "Similarity:",
            "ml_loading_recommendations": "Loading recommendations...",
            "ml_no_recommendations": "No similar theses found",

            # Common
            "na": "N/A",
        },

        "fr": {
            # Header
            "app_title": "🧠 Synapse",
            "app_subtitle": "Votre assistant IA pour explorer les thèses scientifiques",

            # Sidebar
            "sidebar_brand": "🧠 Synapse",
            "sidebar_brand_tagline": "Assistant Scientifique IA",
            "sidebar_status_ready": "Prêt à explorer",
            "sidebar_status_analyzing": "Analyse du document",
            "sidebar_doc_loaded": "Document chargé :",
            "sidebar_new_search": "🔄 Nouvelle recherche",
            "sidebar_history_title": "Conversations récentes",
            "sidebar_history_loading": "Restauration de la conversation...",
            "sidebar_powered_by": "Propulsé par",
            "sidebar_version": "v1.0",

            # Phase 1: Search
            "search_header": "🔍 Explorer les archives HAL",
            "search_input_label": "Entrez votre sujet de recherche ou mots-clés :",
            "search_input_placeholder": "ex : apprentissage automatique, changement climatique, informatique quantique...",
            "search_input_help": "Rechercher des thèses dans la base de données HAL Science",
            "search_num_results": "Nombre de résultats",
            "search_button": "Rechercher",
            "search_searching": "🔎 Recherche dans les archives HAL...",
            "search_no_results": "Aucun résultat trouvé. Essayez d'autres mots-clés.",
            "search_results_found": "{total} thèses trouvées ! Affichage de {shown} résultats.",
            "search_error": "Erreur lors de la recherche : {error}",
            "search_results_header": "📚 Résultats de recherche ({total} au total)",

            # Document card
            "doc_author": "**Auteur :**",
            "doc_date": "**Date :**",
            "doc_domain": "**Domaine :**",
            "doc_keywords": "**Mots-clés :**",
            "doc_abstract": "**Résumé :**",
            "doc_view_hal": "🔗 Voir sur HAL",
            "doc_chat_button": "💬 Discuter avec ce document",
            "doc_pdf_unavailable": "PDF non disponible",

            # Phase 2: Chat
            "chat_header": "💬 Discuter avec le document",
            "chat_current_doc": "📄 Document actuel",
            "chat_doc_title": "**Titre :**",
            "chat_doc_author": "**Auteur :**",
            "chat_doc_date": "**Date :**",
            "chat_doc_view_hal": "🔗 Voir sur HAL",
            "chat_input_placeholder": "Posez une question sur le document...",
            "chat_thinking": "🤔 Réflexion en cours...",
            "chat_sources_header": "📚 Voir les sources",
            "chat_source_chunk": "Source {num} (Chunk {chunk})",
            "chat_error": "Erreur : {error}",
            "chat_show_pdf": "📄 Afficher le PDF",
            "chat_hide_pdf": "📄 Masquer le PDF",
            "chat_pdf_viewer": "📄 Document PDF",

            # Document loading
            "loading_title": "📥 Chargement du document...",
            "loading_downloading": "Téléchargement du PDF...",
            "loading_initializing": "Initialisation du moteur RAG...",
            "loading_processing": "Traitement du document (cela peut prendre une minute)...",
            "loading_success": "✅ Document chargé ! Traité {chunks} chunks ({chars:,} caractères)",
            "loading_error": "Erreur lors du chargement du document : {error}",

            # ML Features
            "ml_filters_header": "🏷️ Filtres Intelligents",
            "ml_domains": "Domaines",
            "ml_methodologies": "Méthodologies",
            "ml_generate_summary": "📝 Générer résumé",
            "ml_generating": "Génération du résumé...",
            "ml_tldr": "**TL;DR**",
            "ml_executive_summary": "**Résumé Exécutif**",
            "ml_detailed_summary": "**Résumé Détaillé**",
            "ml_view_full": "Voir résumé complet",
            "ml_similar_theses": "🔗 Thèses Similaires",
            "ml_similarity": "Similarité :",
            "ml_loading_recommendations": "Chargement des recommandations...",
            "ml_no_recommendations": "Aucune thèse similaire trouvée",

            # Common
            "na": "N/D",
        },
    }

    def __init__(self, language: str = "en"):
        """
        Initialiser les traductions.

        Args:
            language: Code de langue (en ou fr)
        """
        self.language = language if language in self.LANGUAGES else "en"

    def get(self, key: str, **kwargs) -> str:
        """
        Obtenir le texte traduit pour une clé.

        Args:
            key: Clé de traduction
            **kwargs: Paramètres de format pour le formatage de chaîne

        Returns:
            Texte traduit
        """
        text = self.TEXTS.get(self.language, {}).get(key, key)

        if kwargs:
            try:
                return text.format(**kwargs)
            except KeyError:
                return text

        return text

    def set_language(self, language: str):
        """
        Changer la langue actuelle.

        Args:
            language: Code de langue (en ou fr)
        """
        if language in self.LANGUAGES:
            self.language = language

    def get_available_languages(self) -> Dict[str, str]:
        """Obtenir les langues disponibles."""
        return self.LANGUAGES


# Instance singleton
_translations = Translations()


def get_translations() -> Translations:
    """Obtenir l'instance globale de traductions."""
    return _translations


def t(key: str, **kwargs) -> str:
    """
    Fonction raccourcie pour obtenir le texte traduit.

    Args:
        key: Clé de traduction
        **kwargs: Paramètres de format

    Returns:
        Texte traduit
    """
    return _translations.get(key, **kwargs)
