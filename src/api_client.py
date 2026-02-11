"""Client API HAL pour rechercher et récupérer des documents scientifiques."""

import requests
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import logging
from urllib.parse import urljoin
from .config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HALDocument:
    """Représentation structurée d'un document HAL."""

    doc_id: str
    title: str
    abstract: str
    author: str
    pdf_url: Optional[str]
    publication_date: Optional[str]
    keywords: List[str]
    domain: Optional[str]
    url: str

    def to_dict(self) -> Dict[str, Any]:
        """Convertir en dictionnaire pour l'affichage."""
        return {
            "ID": self.doc_id,
            "Title": self.title,
            "Abstract": self.abstract[:300] + "..." if len(self.abstract) > 300 else self.abstract,
            "Author": self.author,
            "PDF": self.pdf_url or "N/A",
            "Date": self.publication_date or "N/A",
            "Keywords": ", ".join(self.keywords[:5]) if self.keywords else "N/A",
            "Domain": self.domain or "N/A",
            "HAL URL": self.url,
        }


class HALAPIClient:
    """Client pour interagir avec l'API HAL Science."""

    def __init__(self):
        """Initialiser le client API HAL."""
        self.base_url = config.HAL_API_BASE_URL
        self.timeout = config.HAL_API_TIMEOUT
        self.results_per_page = config.HAL_RESULTS_PER_PAGE

    def search_theses(
        self,
        query: str,
        rows: int = 10,
        start: int = 0,
        sort: str = "producedDate_tdate desc",
    ) -> Dict[str, Any]:
        """
        Rechercher des thèses dans les archives HAL.

        Args:
            query: Chaîne de recherche
            rows: Nombre de résultats par page
            start: Décalage de départ pour la pagination
            sort: Ordre de tri (par défaut: plus récent en premier)

        Returns:
            Dictionnaire avec 'docs' (liste de HALDocument) et 'numFound' (nombre total de résultats)
        """
        try:
            # Construire les paramètres de requête
            params = {
                "q": query,
                "fq": "docType_s:THESE",  # Filtrer uniquement les thèses
                "rows": rows,
                "start": start,
                "sort": sort,
                "fl": "docid,title_s,abstract_s,authFullName_s,fileMain_s,producedDate_s,keyword_s,domain_s,uri_s",
                "wt": "json",
            }

            logger.info(f"Searching HAL API with query: {query}")
            response = requests.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            response_data = data.get("response", {})

            # Analyser les documents
            docs = []
            for doc in response_data.get("docs", []):
                hal_doc = self._parse_document(doc)
                if hal_doc:
                    docs.append(hal_doc)

            num_found = response_data.get("numFound", 0)
            logger.info(f"Found {num_found} results, returning {len(docs)} documents")

            return {"docs": docs, "numFound": num_found, "start": start}

        except requests.exceptions.Timeout:
            logger.error("HAL API request timed out")
            raise Exception("Request to HAL API timed out. Please try again.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching from HAL API: {e}")
            raise Exception(f"Error connecting to HAL API: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in search_theses: {e}")
            raise

    def _parse_document(self, doc: Dict[str, Any]) -> Optional[HALDocument]:
        """Analyser un document HAL brut en format structuré."""
        try:
            # Extraire les champs avec des valeurs par défaut sûres
            doc_id = doc.get("docid", "")
            title = doc.get("title_s", ["Unknown Title"])[0] if isinstance(doc.get("title_s"), list) else doc.get("title_s", "Unknown Title")
            abstract = doc.get("abstract_s", [""])[0] if isinstance(doc.get("abstract_s"), list) else doc.get("abstract_s", "No abstract available")

            # Gérer les auteurs (peut être une liste)
            authors = doc.get("authFullName_s", [])
            if isinstance(authors, list):
                author = ", ".join(authors[:3])  # 3 premiers auteurs
                if len(authors) > 3:
                    author += " et al."
            else:
                author = authors or "Unknown Author"

            # URL du PDF
            pdf_url = None
            if "fileMain_s" in doc:
                pdf_url = doc["fileMain_s"]
                if not pdf_url.startswith("http"):
                    pdf_url = f"https://hal.science/{doc_id}/document"

            # Autres métadonnées
            publication_date = doc.get("producedDate_s", None)
            keywords = doc.get("keyword_s", [])
            if not isinstance(keywords, list):
                keywords = [keywords] if keywords else []

            domain = doc.get("domain_s", [""])[0] if isinstance(doc.get("domain_s"), list) else doc.get("domain_s", None)
            url = doc.get("uri_s", f"https://hal.science/{doc_id}")

            return HALDocument(
                doc_id=doc_id,
                title=title,
                abstract=abstract,
                author=author,
                pdf_url=pdf_url,
                publication_date=publication_date,
                keywords=keywords,
                domain=domain,
                url=url,
            )

        except Exception as e:
            logger.error(f"Error parsing document: {e}")
            return None

    def download_pdf(self, pdf_url: str, doc_id: str) -> bytes:
        """
        Télécharger le contenu PDF depuis l'URL.

        Args:
            pdf_url: URL du fichier PDF
            doc_id: ID du document pour le logging

        Returns:
            Contenu du PDF en bytes
        """
        try:
            logger.info(f"Downloading PDF for document {doc_id}")
            response = requests.get(
                pdf_url,
                timeout=config.PDF_DOWNLOAD_TIMEOUT,
                headers={"User-Agent": "HAL-RAG-Chatbot/1.0"},
            )
            response.raise_for_status()

            # Vérifier la taille du fichier
            content_length = len(response.content)
            max_size_bytes = config.MAX_PDF_SIZE_MB * 1024 * 1024

            if content_length > max_size_bytes:
                raise Exception(f"PDF file too large: {content_length / (1024*1024):.2f}MB")

            logger.info(f"Successfully downloaded PDF ({content_length / 1024:.2f}KB)")
            return response.content

        except requests.exceptions.Timeout:
            logger.error(f"PDF download timed out for {doc_id}")
            raise Exception("PDF download timed out. The file might be too large.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading PDF: {e}")
            raise Exception(f"Failed to download PDF: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error downloading PDF: {e}")
            raise

    def get_document_metadata(self, doc_id: str) -> Optional[HALDocument]:
        """
        Récupérer les métadonnées d'un document spécifique par ID.

        Args:
            doc_id: ID du document HAL

        Returns:
            Objet HALDocument ou None
        """
        try:
            params = {
                "q": f"docid:{doc_id}",
                "rows": 1,
                "wt": "json",
                "fl": "docid,title_s,abstract_s,authFullName_s,fileMain_s,producedDate_s,keyword_s,domain_s,uri_s",
            }

            response = requests.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            docs = data.get("response", {}).get("docs", [])

            if docs:
                return self._parse_document(docs[0])

            return None

        except Exception as e:
            logger.error(f"Error fetching document metadata for {doc_id}: {e}")
            return None
