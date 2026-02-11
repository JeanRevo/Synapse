"""Fonctions utilitaires pour HAL RAG Chatbot."""

import re
from typing import Optional


def truncate_text(text: str, max_length: int = 300, suffix: str = "...") -> str:
    """
    Tronquer le texte à une longueur maximale.

    Args:
        text: Texte à tronquer
        max_length: Longueur maximale
        suffix: Suffixe à ajouter si tronqué

    Returns:
        Texte tronqué
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)].rstrip() + suffix


def clean_query(query: str) -> str:
    """
    Nettoyer et normaliser la requête de recherche.

    Args:
        query: Requête de recherche brute

    Returns:
        Requête nettoyée
    """
    # Supprimer les espaces supplémentaires
    query = " ".join(query.split())

    # Supprimer les caractères spéciaux qui pourraient casser l'API
    query = re.sub(r'[^\w\s\-"\']', " ", query)

    return query.strip()


def format_file_size(size_bytes: int) -> str:
    """
    Formater la taille de fichier dans un format lisible.

    Args:
        size_bytes: Taille en bytes

    Returns:
        Chaîne de taille formatée
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def validate_pdf_url(url: Optional[str]) -> bool:
    """
    Valider si une URL est une URL PDF valide.

    Args:
        url: URL à valider

    Returns:
        True si valide, False sinon
    """
    if not url:
        return False

    url = url.lower()
    return url.startswith("http") and (".pdf" in url or "/document" in url)


def highlight_keywords(text: str, keywords: list) -> str:
    """
    Surligner les mots-clés dans le texte (à des fins d'affichage).

    Args:
        text: Texte à traiter
        keywords: Liste de mots-clés à surligner

    Returns:
        Texte avec mots-clés surlignés
    """
    for keyword in keywords:
        if keyword and len(keyword) > 2:
            # Remplacement insensible à la casse
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            text = pattern.sub(f"**{keyword}**", text)

    return text
