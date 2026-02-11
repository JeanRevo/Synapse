"""Système de recommandation de thèses basé sur la similarité."""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThesisRecommender:
    """Recommander des thèses similaires basées sur la similarité sémantique."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialiser le système de recommandation.

        Args:
            model_name: Modèle SentenceTransformer à utiliser
        """
        try:
            logger.info(f"Loading recommendation model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.thesis_embeddings = {}
            self.thesis_metadata = {}
            logger.info("Recommendation model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def index_thesis(self, thesis_id: str, text: str, metadata: Dict[str, Any]):
        """
        Indexer une thèse pour les recommandations.

        Args:
            thesis_id: Identifiant unique de la thèse
            text: Résumé de la thèse ou texte complet
            metadata: Métadonnées de la thèse (titre, auteur, etc.)
        """
        try:
            # Générer l'embedding
            embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)

            # Stocker
            self.thesis_embeddings[thesis_id] = embedding
            self.thesis_metadata[thesis_id] = metadata

            logger.info(f"Indexed thesis: {thesis_id}")

        except Exception as e:
            logger.error(f"Error indexing thesis {thesis_id}: {e}")

    def index_multiple(self, theses: List[Dict[str, Any]]):
        """
        Indexer plusieurs thèses à la fois.

        Args:
            theses: Liste de dictionnaires de thèses avec 'id', 'text' et 'metadata'
        """
        logger.info(f"Indexing {len(theses)} theses...")

        for thesis in theses:
            self.index_thesis(
                thesis['id'],
                thesis['text'],
                thesis.get('metadata', {})
            )

        logger.info(f"Indexed {len(self.thesis_embeddings)} theses total")

    def recommend(
        self,
        thesis_id: str,
        top_k: int = 5,
        filters: Optional[Dict[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Recommander des thèses similaires.

        Args:
            thesis_id: ID de la thèse pour laquelle trouver des similaires
            top_k: Nombre de recommandations à retourner
            filters: Filtres optionnels (domaines, méthodologies, etc.)

        Returns:
            Liste de thèses recommandées avec scores de similarité
        """
        if thesis_id not in self.thesis_embeddings:
            logger.warning(f"Thesis {thesis_id} not indexed")
            return []

        # Obtenir l'embedding de la requête
        query_embedding = self.thesis_embeddings[thesis_id]

        # Calculer les similarités avec toutes les autres thèses
        similarities = []

        for tid, embedding in self.thesis_embeddings.items():
            if tid == thesis_id:  # Ignorer soi-même
                continue

            # Appliquer les filtres si fournis
            if filters and not self._matches_filters(tid, filters):
                continue

            # Calculer la similarité cosinus
            sim = cosine_similarity([query_embedding], [embedding])[0][0]

            similarities.append({
                "thesis_id": tid,
                "similarity": float(sim),
                "metadata": self.thesis_metadata[tid],
            })

        # Trier par similarité et retourner le top-k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

    def recommend_by_text(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Recommander des thèses basées sur une requête texte.

        Args:
            query_text: Requête de recherche
            top_k: Nombre de recommandations
            filters: Filtres optionnels

        Returns:
            Liste de thèses recommandées
        """
        # Encoder la requête
        query_embedding = self.model.encode(
            query_text,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Calculer les similarités
        similarities = []

        for tid, embedding in self.thesis_embeddings.items():
            # Appliquer les filtres
            if filters and not self._matches_filters(tid, filters):
                continue

            sim = cosine_similarity([query_embedding], [embedding])[0][0]

            similarities.append({
                "thesis_id": tid,
                "similarity": float(sim),
                "metadata": self.thesis_metadata[tid],
            })

        # Trier et retourner le top-k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

    def _matches_filters(self, thesis_id: str, filters: Dict[str, List[str]]) -> bool:
        """Vérifier si une thèse correspond aux filtres donnés."""
        metadata = self.thesis_metadata.get(thesis_id, {})

        for filter_key, filter_values in filters.items():
            if filter_key not in metadata:
                continue

            # Vérifier si l'une des valeurs de filtre correspond
            thesis_values = metadata[filter_key]
            if isinstance(thesis_values, str):
                thesis_values = [thesis_values]

            if not any(fv in thesis_values for fv in filter_values):
                return False

        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Obtenir les statistiques du système de recommandation."""
        return {
            "total_theses_indexed": len(self.thesis_embeddings),
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "model_name": self.model._modules['0'].auto_model.config.name_or_path,
        }

    def find_clusters(self, n_clusters: int = 5) -> Dict[str, List[str]]:
        """
        Regrouper les thèses en groupes.

        Args:
            n_clusters: Nombre de clusters

        Returns:
            Dictionnaire mappant les IDs de cluster aux IDs de thèse
        """
        if len(self.thesis_embeddings) < n_clusters:
            logger.warning("Not enough theses for clustering")
            return {}

        try:
            from sklearn.cluster import KMeans

            # Obtenir tous les embeddings
            thesis_ids = list(self.thesis_embeddings.keys())
            embeddings = np.array([self.thesis_embeddings[tid] for tid in thesis_ids])

            # Effectuer le clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embeddings)

            # Grouper par cluster
            clusters = {}
            for i, label in enumerate(labels):
                cluster_id = f"cluster_{label}"
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(thesis_ids[i])

            logger.info(f"Created {n_clusters} clusters")
            return clusters

        except Exception as e:
            logger.error(f"Error clustering: {e}")
            return {}

    def save_index(self, filepath: str):
        """Sauvegarder l'index sur le disque."""
        import pickle

        try:
            data = {
                "embeddings": self.thesis_embeddings,
                "metadata": self.thesis_metadata,
            }
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved index to {filepath}")
        except Exception as e:
            logger.error(f"Error saving index: {e}")

    def load_index(self, filepath: str):
        """Charger l'index depuis le disque."""
        import pickle

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            self.thesis_embeddings = data["embeddings"]
            self.thesis_metadata = data["metadata"]

            logger.info(f"Loaded index from {filepath}")
            logger.info(f"Loaded {len(self.thesis_embeddings)} theses")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
