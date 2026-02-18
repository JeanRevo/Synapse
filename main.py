"""
Synapse - Backend FastAPI
API REST pour le pipeline RAG d'analyse de thèses scientifiques.
"""

import sys
import logging
import base64
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ajouter src au chemin
sys.path.insert(0, str(Path(__file__).parent))

from src.api_client import HALAPIClient
from src.rag_engine import RAGEngine
from src.utils import clean_query, validate_pdf_url
from src.config import config
from src.pdf_annotator import PDFAnnotator

# ML Features
ml_classifier = None
ml_summarizer = None
ml_recommender = None

if config.ENABLE_CLASSIFICATION:
    from src.ml_features.classifier import ThesisClassifier
    ml_classifier = ThesisClassifier(use_ml_model=config.USE_ML_MODELS)

if config.ENABLE_SUMMARIZATION:
    from src.ml_features.summarizer import ThesisSummarizer
    ml_summarizer = ThesisSummarizer(use_transformers=config.USE_ML_MODELS)

if config.ENABLE_RECOMMENDATIONS:
    from src.ml_features.recommender import ThesisRecommender
    ml_recommender = ThesisRecommender()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Synapse API",
    description="API REST pour l'analyse de thèses scientifiques via RAG",
    version="2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir les fichiers statiques
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ============================================================
# State (in-memory, single user)
# ============================================================

hal_client = HALAPIClient()
rag_engine: Optional[RAGEngine] = None
current_doc: Optional[dict] = None
pdf_bytes_store: Optional[bytes] = None
search_results_cache: list = []
conversation_history: list = []  # Historique des conversations
chat_messages: list = []  # Messages de la conversation en cours
last_sources: list = []  # Dernières sources pour le surlignage PDF
last_answer: Optional[str] = None  # Dernière réponse pour le surlignage


# ============================================================
# Pydantic Models
# ============================================================

class SearchRequest(BaseModel):
    query: str
    num_results: int = 10
    start: int = 0


class AskRequest(BaseModel):
    question: str


class LoadDocRequest(BaseModel):
    doc_index: int


class RestoreRequest(BaseModel):
    conv_index: int


class SummarizeRequest(BaseModel):
    doc_index: int


# ============================================================
# Routes - Pages
# ============================================================

@app.get("/")
async def serve_landing():
    """Servir la landing page."""
    return FileResponse(Path(__file__).parent / "landing.html")


@app.get("/app")
async def serve_app():
    """Servir l'interface principale."""
    return FileResponse(Path(__file__).parent / "static" / "app.html")


# ============================================================
# Routes - API
# ============================================================

@app.post("/api/search")
async def search_theses(req: SearchRequest):
    """Rechercher des thèses sur HAL."""
    global search_results_cache

    try:
        cleaned = clean_query(req.query)
        results = hal_client.search_theses(query=cleaned, rows=req.num_results, start=req.start)

        docs = []
        for doc in results.get("docs", []):
            doc_dict = {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "author": doc.author,
                "abstract": doc.abstract[:400] + "..." if len(doc.abstract) > 400 else doc.abstract,
                "full_abstract": doc.abstract,
                "pdf_url": doc.pdf_url,
                "publication_date": doc.publication_date,
                "keywords": doc.keywords[:5] if doc.keywords else [],
                "domain": doc.domain,
                "url": doc.url,
                "has_pdf": validate_pdf_url(doc.pdf_url),
                "classification": None,
            }

            # ML Classification
            if ml_classifier:
                try:
                    classification = ml_classifier.classify(doc.abstract)
                    doc_dict["classification"] = classification
                except Exception:
                    pass

            # Indexer pour recommandations
            if ml_recommender:
                try:
                    ml_recommender.index_thesis(
                        doc.doc_id, doc.abstract,
                        {"title": doc.title, "author": doc.author, "domains": doc.domain}
                    )
                except Exception:
                    pass

            docs.append(doc_dict)

        # Nouvelle recherche : remplacer le cache. Pagination : ajouter au cache.
        if req.start == 0:
            search_results_cache = docs
        else:
            search_results_cache.extend(docs)

        return {
            "success": True,
            "num_found": results["numFound"],
            "start": req.start,
            "docs": docs,
        }

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/load")
async def load_document(req: LoadDocRequest):
    """Charger et vectoriser un document pour le chat."""
    global rag_engine, current_doc, pdf_bytes_store, chat_messages, last_sources, last_answer

    try:
        if req.doc_index < 0 or req.doc_index >= len(search_results_cache):
            raise HTTPException(status_code=400, detail="Invalid document index")

        doc_data = search_results_cache[req.doc_index]

        if not doc_data.get("has_pdf"):
            raise HTTPException(status_code=400, detail="No PDF available")

        # Sauvegarder la conversation en cours
        _save_current_conversation()

        # Télécharger le PDF
        logger.info(f"Downloading PDF for {doc_data['doc_id']}...")
        pdf_bytes_store = hal_client.download_pdf(doc_data["pdf_url"], doc_data["doc_id"])

        # Initialiser le RAG engine si nécessaire
        if rag_engine is None:
            rag_engine = RAGEngine()
        else:
            rag_engine.reset()

        # Ingérer le document
        doc_metadata = {
            "title": doc_data["title"],
            "author": doc_data["author"],
            "doc_id": doc_data["doc_id"],
        }

        stats = rag_engine.ingest_document(pdf_bytes_store, doc_data["doc_id"], doc_metadata)
        current_doc = doc_data
        chat_messages = []
        last_sources = []
        last_answer = None

        return {
            "success": True,
            "doc_id": doc_data["doc_id"],
            "title": doc_data["title"],
            "author": doc_data["author"],
            "stats": {
                "total_chunks": stats.get("total_chunks", 0),
                "total_characters": stats.get("total_characters", 0),
                "chunking_method": stats.get("chunking_method", "flat"),
                "chapter_chunks": stats.get("chapter_chunks", 0),
                "section_chunks": stats.get("section_chunks", 0),
                "paragraph_chunks": stats.get("paragraph_chunks", 0),
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Load error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask")
async def ask_question(req: AskRequest):
    """Poser une question sur le document chargé."""
    global last_sources, last_answer, chat_messages

    if rag_engine is None or current_doc is None:
        raise HTTPException(status_code=400, detail="No document loaded")

    try:
        response = rag_engine.ask_question(req.question)

        # Stocker pour surlignage et historique
        last_sources = response["sources"]
        last_answer = response["answer"]
        chat_messages.append({"role": "user", "content": req.question})
        chat_messages.append({"role": "assistant", "content": response["answer"], "sources": response["sources"]})

        return {
            "success": True,
            "answer": response["answer"],
            "sources": response["sources"],
            "index_used": response.get("index_used", "default"),
        }

    except Exception as e:
        logger.error(f"Ask error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pdf")
async def get_pdf():
    """Récupérer le PDF original en base64."""
    if pdf_bytes_store is None:
        raise HTTPException(status_code=404, detail="No PDF loaded")

    return {
        "pdf_base64": base64.b64encode(pdf_bytes_store).decode("utf-8"),
        "doc": current_doc,
    }


@app.get("/api/pdf/highlighted")
async def get_highlighted_pdf():
    """Récupérer le PDF avec surlignages des sources."""
    if pdf_bytes_store is None:
        raise HTTPException(status_code=404, detail="No PDF loaded")

    if not last_sources:
        return {
            "pdf_base64": base64.b64encode(pdf_bytes_store).decode("utf-8"),
            "highlighted": False,
            "sources_count": 0,
        }

    try:
        highlighted_pdf = PDFAnnotator.generate_highlighted_pdf(
            pdf_bytes_store, last_sources, max_sources=8, answer_text=last_answer
        )

        pages_with_sources = sorted(set(s.get("page_num", 1) for s in last_sources))

        return {
            "pdf_base64": base64.b64encode(highlighted_pdf).decode("utf-8"),
            "highlighted": True,
            "sources_count": len(last_sources),
            "pages_with_sources": pages_with_sources,
            "colors": PDFAnnotator.COLORS[:len(last_sources)],
        }
    except Exception as e:
        logger.error(f"PDF highlight error: {e}")
        return {
            "pdf_base64": base64.b64encode(pdf_bytes_store).decode("utf-8"),
            "highlighted": False,
            "sources_count": 0,
        }


@app.get("/api/pdf/download")
async def download_highlighted_pdf():
    """Télécharger le PDF avec surlignages."""
    if pdf_bytes_store is None:
        raise HTTPException(status_code=404, detail="No PDF loaded")

    try:
        if last_sources:
            pdf_data = PDFAnnotator.generate_highlighted_pdf(
                pdf_bytes_store, last_sources, max_sources=8, answer_text=last_answer
            )
        else:
            pdf_data = pdf_bytes_store

        return Response(
            content=pdf_data,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=synapse_annotated.pdf"}
        )
    except Exception:
        return Response(
            content=pdf_bytes_store,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=synapse_document.pdf"}
        )


# ============================================================
# Historique des conversations
# ============================================================

def _save_current_conversation():
    """Sauvegarder la conversation en cours dans l'historique."""
    global conversation_history, chat_messages, current_doc

    if not current_doc or not chat_messages:
        return

    user_msgs = [m for m in chat_messages if m["role"] == "user"]
    if not user_msgs:
        return

    # Vérifier si on met à jour une conversation existante
    existing_idx = None
    for i, conv in enumerate(conversation_history):
        if conv.get("doc_id") == current_doc.get("doc_id"):
            existing_idx = i
            break

    conv_data = {
        "doc_id": current_doc["doc_id"],
        "title": current_doc["title"],
        "author": current_doc["author"],
        "pdf_url": current_doc.get("pdf_url"),
        "url": current_doc.get("url"),
        "full_abstract": current_doc.get("full_abstract", ""),
        "messages": list(chat_messages),
        "message_count": len(user_msgs),
        "timestamp": datetime.now().strftime("%H:%M"),
    }

    if existing_idx is not None:
        conversation_history[existing_idx] = conv_data
    else:
        conversation_history.insert(0, conv_data)

    conversation_history[:] = conversation_history[:15]


@app.get("/api/history")
async def get_history():
    """Récupérer l'historique des conversations."""
    # Sauvegarder la conversation en cours d'abord
    _save_current_conversation()

    return {
        "conversations": [
            {
                "index": i,
                "doc_id": conv["doc_id"],
                "title": conv["title"],
                "author": conv["author"],
                "message_count": conv["message_count"],
                "timestamp": conv["timestamp"],
                "is_active": current_doc and current_doc.get("doc_id") == conv["doc_id"],
            }
            for i, conv in enumerate(conversation_history)
        ]
    }


@app.post("/api/history/restore")
async def restore_conversation(req: RestoreRequest):
    """Restaurer une conversation depuis l'historique."""
    global rag_engine, current_doc, pdf_bytes_store, chat_messages, last_sources, last_answer

    if req.conv_index < 0 or req.conv_index >= len(conversation_history):
        raise HTTPException(status_code=400, detail="Invalid conversation index")

    # Sauvegarder la conversation en cours
    _save_current_conversation()

    conv = conversation_history[req.conv_index]

    try:
        # Télécharger le PDF
        pdf_bytes_store = hal_client.download_pdf(conv["pdf_url"], conv["doc_id"])

        # Ré-ingérer le document
        if rag_engine is None:
            rag_engine = RAGEngine()
        else:
            rag_engine.reset()

        doc_metadata = {
            "title": conv["title"],
            "author": conv["author"],
            "doc_id": conv["doc_id"],
        }
        rag_engine.ingest_document(pdf_bytes_store, conv["doc_id"], doc_metadata)

        # Restaurer l'état
        current_doc = {
            "doc_id": conv["doc_id"],
            "title": conv["title"],
            "author": conv["author"],
            "pdf_url": conv["pdf_url"],
            "url": conv.get("url"),
            "full_abstract": conv.get("full_abstract", ""),
        }
        chat_messages = conv["messages"]
        last_sources = []
        last_answer = None

        return {
            "success": True,
            "title": conv["title"],
            "author": conv["author"],
            "messages": conv["messages"],
        }

    except Exception as e:
        logger.error(f"Restore error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ML Features
# ============================================================

@app.post("/api/summarize")
async def summarize_thesis(req: SummarizeRequest):
    """Générer un résumé pour une thèse."""
    if not ml_summarizer:
        raise HTTPException(status_code=400, detail="Summarization not enabled")

    if req.doc_index < 0 or req.doc_index >= len(search_results_cache):
        raise HTTPException(status_code=400, detail="Invalid document index")

    doc = search_results_cache[req.doc_index]

    try:
        summary = ml_summarizer.generate_summaries(doc["full_abstract"], doc["title"])
        return {"success": True, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/recommendations")
async def get_recommendations():
    """Obtenir les thèses similaires au document chargé."""
    if not ml_recommender or not current_doc:
        return {"recommendations": []}

    try:
        similar = ml_recommender.recommend_by_text(
            current_doc.get("full_abstract", ""),
            top_k=3,
            exclude_thesis_id=current_doc.get("doc_id"),
        )

        recs = []
        for rec in (similar or []):
            rec_doc_id = rec.get("thesis_id", "")
            rec_data = {
                "title": rec["metadata"].get("title", "N/A"),
                "author": rec["metadata"].get("author", ""),
                "similarity": round(rec["similarity"], 2),
                "doc_id": rec_doc_id,
                "has_pdf": False,
                "search_index": None,
            }
            # Retrouver dans le cache de recherche pour pouvoir charger
            for idx, cached in enumerate(search_results_cache):
                if cached.get("doc_id") == rec_doc_id:
                    rec_data["has_pdf"] = cached.get("has_pdf", False)
                    rec_data["search_index"] = idx
                    break
            recs.append(rec_data)

        return {"recommendations": recs}
    except Exception:
        return {"recommendations": []}


# ============================================================
# Status & Reset
# ============================================================

@app.get("/api/status")
async def get_status():
    """Statut de l'application."""
    return {
        "rag_loaded": rag_engine is not None and current_doc is not None,
        "current_doc": current_doc.get("title") if current_doc else None,
        "current_doc_id": current_doc.get("doc_id") if current_doc else None,
        "history_count": len(conversation_history),
        "ml_features": {
            "classification": ml_classifier is not None,
            "summarization": ml_summarizer is not None,
            "recommendations": ml_recommender is not None,
        }
    }


@app.post("/api/reset")
async def reset_app():
    """Réinitialiser le RAG engine."""
    global rag_engine, current_doc, pdf_bytes_store, chat_messages, last_sources, last_answer
    _save_current_conversation()
    if rag_engine:
        rag_engine.reset()
    current_doc = None
    pdf_bytes_store = None
    chat_messages = []
    last_sources = []
    last_answer = None
    return {"success": True}
