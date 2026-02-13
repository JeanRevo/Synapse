"""
Synapse - Assistant Scientifique IA

Application d'analyse et de discussion avec les thèses scientifiques via les archives HAL Science.
"""

import streamlit as st
import pandas as pd
from typing import Optional, List, Dict, Any
import sys
from pathlib import Path
import base64
from datetime import datetime

# Ajouter src au chemin
sys.path.insert(0, str(Path(__file__).parent))

from src.api_client import HALAPIClient, HALDocument
from src.rag_engine import RAGEngine
from src.utils import truncate_text, clean_query, validate_pdf_url
from src.config import config
from src.translations import get_translations, t
from src.pdf_annotator import PDFAnnotator

# Importer streamlit-pdf-viewer
try:
    from streamlit_pdf_viewer import pdf_viewer
except ImportError:
    pdf_viewer = None  # Fallback si non installé

# Fonctionnalités ML
if config.ENABLE_CLASSIFICATION:
    from src.ml_features.classifier import ThesisClassifier
if config.ENABLE_SUMMARIZATION:
    from src.ml_features.summarizer import ThesisSummarizer
if config.ENABLE_RECOMMENDATIONS:
    from src.ml_features.recommender import ThesisRecommender

# Configuration de la page
st.set_page_config(
    page_title="Synapse - Assistant Scientifique",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personnalisé
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .ml-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        margin: 0.2rem;
        border-radius: 0.3rem;
        background-color: #e3f2fd;
        color: #1976d2;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Initialiser l'état de session
def init_session_state():
    """Initialiser les variables d'état de session Streamlit."""
    if "app_state" not in st.session_state:
        st.session_state.app_state = "search"
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "selected_doc" not in st.session_state:
        st.session_state.selected_doc = None
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None
    if "document_loaded" not in st.session_state:
        st.session_state.document_loaded = False
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "hal_client" not in st.session_state:
        st.session_state.hal_client = HALAPIClient()
    if "language" not in st.session_state:
        st.session_state.language = "fr"
    if "pdf_bytes" not in st.session_state:
        st.session_state.pdf_bytes = None
    if "show_pdf" not in st.session_state:
        st.session_state.show_pdf = True
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1
    if "current_annotations" not in st.session_state:
        st.session_state.current_annotations = []
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = None
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []  # Liste des conversations passées
    if "active_conversation_id" not in st.session_state:
        st.session_state.active_conversation_id = None

    # Fonctionnalités ML initialization
    if config.ENABLE_CLASSIFICATION and "ml_classifier" not in st.session_state:
        with st.spinner("Initialisation de la classification ML..."):
            st.session_state.ml_classifier = ThesisClassifier(use_ml_model=config.USE_ML_MODELS)

    if config.ENABLE_SUMMARIZATION and "ml_summarizer" not in st.session_state:
        with st.spinner("Initialisation du résumé automatique..."):
            st.session_state.ml_summarizer = ThesisSummarizer(use_transformers=config.USE_ML_MODELS)

    if config.ENABLE_RECOMMENDATIONS and "ml_recommender" not in st.session_state:
        with st.spinner("Initialisation des recommandations..."):
            st.session_state.ml_recommender = ThesisRecommender()

    # ML state
    if "selected_domains" not in st.session_state:
        st.session_state.selected_domains = []
    if "selected_methodologies" not in st.session_state:
        st.session_state.selected_methodologies = []
    if "show_summaries" not in st.session_state:
        st.session_state.show_summaries = {}


def render_header():
    """Afficher l'en-tête de l'application."""
    translations = get_translations()
    st.markdown(f'<div class="main-header">{translations.get("app_title")}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-header">{translations.get("app_subtitle")}</div>',
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Afficher la barre latérale Synapse."""
    translations = get_translations()

    with st.sidebar:
        # Branding Synapse
        st.markdown(
            """
            <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
                <div style="font-size: 2rem; font-weight: bold; color: #1f77b4;">
                    🧠 Synapse
                </div>
                <div style="font-size: 0.9rem; color: #888; margin-top: 0.2rem;">
                    {tagline}
                </div>
            </div>
            """.format(tagline=translations.get("sidebar_brand_tagline")),
            unsafe_allow_html=True,
        )

        # Indicateur de statut
        if st.session_state.document_loaded:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
                    border-radius: 8px; padding: 0.7rem; margin: 0.5rem 0;
                    border-left: 4px solid #4caf50;
                ">
                    <div style="font-size: 0.8rem; color: #2e7d32; font-weight: 600;">
                        💬 {translations.get("sidebar_status_analyzing")}
                    </div>
                    <div style="font-size: 0.75rem; color: #555; margin-top: 0.3rem;">
                        {st.session_state.selected_doc.title[:60]}...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
                    border-radius: 8px; padding: 0.7rem; margin: 0.5rem 0;
                    border-left: 4px solid #1976d2;
                ">
                    <div style="font-size: 0.8rem; color: #1565c0; font-weight: 600;">
                        🔍 {translations.get("sidebar_status_ready")}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Historique des conversations
        if st.session_state.conversation_history:
            st.divider()
            st.markdown(
                f"<div style='font-size: 0.85rem; font-weight: 600; color: #555; margin-bottom: 0.3rem;'>"
                f"💬 {translations.get('sidebar_history_title')}</div>",
                unsafe_allow_html=True,
            )

            for i, conv in enumerate(st.session_state.conversation_history):
                is_active = (
                    st.session_state.document_loaded
                    and st.session_state.selected_doc
                    and st.session_state.selected_doc.doc_id == conv["doc_id"]
                )
                title_short = conv["title"][:45] + "..." if len(conv["title"]) > 45 else conv["title"]
                msg_count = conv.get("message_count", 0)
                time_str = conv.get("timestamp", "")

                if is_active:
                    st.markdown(
                        f"""<div style="
                            background: #e3f2fd; border-radius: 6px; padding: 0.4rem 0.6rem;
                            margin: 0.2rem 0; border-left: 3px solid #1976d2;
                            font-size: 0.78rem;
                        ">
                            <div style="font-weight: 600; color: #1565c0;">📄 {title_short}</div>
                            <div style="color: #888; font-size: 0.7rem;">{msg_count} messages · {time_str}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    if st.button(
                        f"📄 {title_short}",
                        key=f"conv_hist_{i}",
                        use_container_width=True,
                        help=f"{msg_count} messages · {time_str}",
                    ):
                        with st.spinner(translations.get("sidebar_history_loading")):
                            _restore_conversation(i)
                            st.rerun()

        # Filtres intelligents ML (uniquement en phase de recherche)
        if config.ENABLE_CLASSIFICATION and st.session_state.app_state == "search" and st.session_state.search_results:
            st.divider()
            st.subheader(translations.get("ml_filters_header"))

            docs = st.session_state.search_results.get("docs", [])
            all_domains = set()
            all_methods = set()

            for doc in docs:
                if hasattr(doc, 'ml_classification'):
                    all_domains.update(doc.ml_classification.get('domains', []))
                    all_methods.update(doc.ml_classification.get('methodologies', []))

            if all_domains:
                st.session_state.selected_domains = st.multiselect(
                    translations.get("ml_domains"),
                    sorted(list(all_domains))
                )

            if all_methods:
                st.session_state.selected_methodologies = st.multiselect(
                    translations.get("ml_methodologies"),
                    sorted(list(all_methods))
                )

        # Recommandations (uniquement en phase de chat)
        if config.ENABLE_RECOMMENDATIONS and st.session_state.app_state == "chat" and st.session_state.document_loaded:
            st.divider()
            st.subheader(translations.get("ml_similar_theses"))

            if hasattr(st.session_state.selected_doc, 'doc_id'):
                try:
                    similar = st.session_state.ml_recommender.recommend_by_text(
                        st.session_state.selected_doc.abstract,
                        top_k=3,
                        exclude_thesis_id=st.session_state.selected_doc.doc_id
                    )

                    if similar:
                        for i, rec in enumerate(similar, 1):
                            with st.container():
                                st.write(f"**{i}. {rec['metadata'].get('title', 'N/A')[:60]}...**")
                                st.caption(f"{translations.get('ml_similarity')} {rec['similarity']:.0%}")
                    else:
                        st.info(translations.get("ml_no_recommendations"))
                except:
                    pass

        # Bouton nouvelle recherche
        st.divider()
        if st.button(translations.get("sidebar_new_search"), use_container_width=True):
            reset_application()
            st.rerun()

        # Footer
        st.markdown(
            f"""
            <div style="
                text-align: center; padding: 1rem 0 0.5rem 0;
                color: #999; font-size: 0.75rem;
            ">
                {translations.get("sidebar_powered_by")}
                <a href="https://hal.science" target="_blank" style="color: #1976d2; text-decoration: none;">
                    HAL Science
                </a>
                <br/>
                <span style="font-size: 0.7rem;">Synapse {translations.get("sidebar_version")}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def classify_and_filter_results():
    """Classifier les résultats de recherche et appliquer les filtres."""
    if not config.ENABLE_CLASSIFICATION or not st.session_state.search_results:
        return

    docs = st.session_state.search_results.get("docs", [])

    # Classifier tous les documents
    for doc in docs:
        if not hasattr(doc, 'ml_classification'):
            classification = st.session_state.ml_classifier.classify(doc.abstract)
            doc.ml_classification = classification

    # Appliquer les filtres
    if st.session_state.selected_domains or st.session_state.selected_methodologies:
        filtered_docs = []
        for doc in docs:
            match = True
            if st.session_state.selected_domains:
                if not any(d in doc.ml_classification.get('domains', []) for d in st.session_state.selected_domains):
                    match = False
            if st.session_state.selected_methodologies:
                if not any(m in doc.ml_classification.get('methodologies', []) for m in st.session_state.selected_methodologies):
                    match = False
            if match:
                filtered_docs.append(doc)

        st.session_state.search_results["docs"] = filtered_docs


def render_search_phase():
    """Afficher la phase de recherche et découverte."""
    translations = get_translations()
    st.header(translations.get("search_header"))

    # Formulaire de recherche
    with st.form("search_form"):
        query = st.text_input(
            translations.get("search_input_label"),
            placeholder=translations.get("search_input_placeholder"),
            help=translations.get("search_input_help"),
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            num_results = st.slider(translations.get("search_num_results"), min_value=5, max_value=50, value=10, step=5)
        with col2:
            search_button = st.form_submit_button(translations.get("search_button"), use_container_width=True)

    # Gérer la recherche
    if search_button and query:
        with st.spinner(translations.get("search_searching")):
            try:
                cleaned_query = clean_query(query)
                results = st.session_state.hal_client.search_theses(query=cleaned_query, rows=num_results)
                st.session_state.search_results = results

                # Indexer pour les recommandations
                if config.ENABLE_RECOMMENDATIONS:
                    for doc in results.get("docs", []):
                        st.session_state.ml_recommender.index_thesis(
                            doc.doc_id,
                            doc.abstract,
                            {"title": doc.title, "author": doc.author, "domains": doc.domain}
                        )

                if results["numFound"] == 0:
                    st.warning(translations.get("search_no_results"))
                else:
                    st.success(translations.get("search_results_found", total=results['numFound'], shown=len(results['docs'])))

            except Exception as e:
                st.error(translations.get("search_error", error=str(e)))

    # Classifier et filtrer
    classify_and_filter_results()

    # Afficher les résultats
    if st.session_state.search_results and st.session_state.search_results["docs"]:
        st.divider()

        filtered_count = len(st.session_state.search_results["docs"])
        st.subheader(translations.get("search_results_header", total=filtered_count))

        for idx, doc in enumerate(st.session_state.search_results["docs"]):
            with st.expander(f"📄 {idx + 1}. {doc.title}", expanded=False):
                # Badges de classification ML
                if config.ENABLE_CLASSIFICATION and hasattr(doc, 'ml_classification'):
                    badges_html = ""
                    for domain in doc.ml_classification.get('domains', [])[:2]:
                        badges_html += f'<span class="ml-badge">🏷️ {domain}</span>'
                    for method in doc.ml_classification.get('methodologies', [])[:2]:
                        badges_html += f'<span class="ml-badge">📊 {method}</span>'
                    if badges_html:
                        st.markdown(badges_html, unsafe_allow_html=True)

                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"{translations.get('doc_author')} {doc.author}")
                    st.markdown(f"{translations.get('doc_date')} {doc.publication_date or translations.get('na')}")
                    st.markdown(f"{translations.get('doc_domain')} {doc.domain or translations.get('na')}")

                    if doc.keywords:
                        st.markdown(f"{translations.get('doc_keywords')} {', '.join(doc.keywords[:5])}")

                    st.markdown(translations.get("doc_abstract"))
                    st.write(truncate_text(doc.abstract, 400))

                    # Bouton de résumé ML
                    if config.ENABLE_SUMMARIZATION:
                        summary_key = f"summary_{doc.doc_id}"
                        if st.button(translations.get("ml_generate_summary"), key=f"btn_{idx}"):
                            with st.spinner(translations.get("ml_generating")):
                                summary = st.session_state.ml_summarizer.generate_summaries(doc.abstract, doc.title)
                                st.session_state.show_summaries[summary_key] = summary

                        if summary_key in st.session_state.show_summaries:
                            summary = st.session_state.show_summaries[summary_key]
                            st.success(translations.get("ml_tldr"))
                            st.write(summary['tldr'])

                    st.markdown(f"[{translations.get('doc_view_hal')}]({doc.url})")

                with col2:
                    if validate_pdf_url(doc.pdf_url):
                        if st.button(translations.get("doc_chat_button"), key=f"select_{idx}"):
                            load_document_for_chat(doc)
                    else:
                        st.warning(translations.get("doc_pdf_unavailable"))


def _save_current_conversation():
    """Sauvegarder la conversation en cours dans l'historique (si elle existe)."""
    if not st.session_state.document_loaded or not st.session_state.selected_doc:
        return
    if not st.session_state.chat_messages:
        return  # Pas de messages, rien à sauvegarder

    doc = st.session_state.selected_doc
    conv_id = f"{doc.doc_id}_{datetime.now().strftime('%H%M%S')}"

    # Vérifier si on met à jour une conversation existante ou on en crée une nouvelle
    existing_idx = None
    for i, conv in enumerate(st.session_state.conversation_history):
        if conv.get("doc_id") == doc.doc_id:
            existing_idx = i
            break

    conv_data = {
        "id": conv_id,
        "doc_id": doc.doc_id,
        "title": doc.title,
        "author": doc.author,
        "pdf_url": doc.pdf_url,
        "url": doc.url,
        "abstract": doc.abstract,
        "messages": list(st.session_state.chat_messages),
        "message_count": len([m for m in st.session_state.chat_messages if m["role"] == "user"]),
        "timestamp": datetime.now().strftime("%H:%M"),
    }

    if existing_idx is not None:
        st.session_state.conversation_history[existing_idx] = conv_data
    else:
        st.session_state.conversation_history.insert(0, conv_data)

    # Garder max 15 conversations
    st.session_state.conversation_history = st.session_state.conversation_history[:15]


def _restore_conversation(conv_index: int):
    """Restaurer une conversation depuis l'historique."""
    conv = st.session_state.conversation_history[conv_index]

    # Sauvegarder la conversation actuelle d'abord
    _save_current_conversation()

    # Re-télécharger le PDF et re-ingérer le document
    try:
        pdf_bytes = st.session_state.hal_client.download_pdf(conv["pdf_url"], conv["doc_id"])

        if st.session_state.rag_engine is None:
            st.session_state.rag_engine = RAGEngine()

        # Recréer un objet HALDocument minimal
        restored_doc = HALDocument(
            doc_id=conv["doc_id"],
            title=conv["title"],
            author=conv["author"],
            abstract=conv["abstract"],
            pdf_url=conv["pdf_url"],
            url=conv["url"],
            publication_date=None,
            keywords=[],
            domain=None,
        )

        doc_metadata = {"title": conv["title"], "author": conv["author"], "doc_id": conv["doc_id"]}
        st.session_state.rag_engine.ingest_document(pdf_bytes, conv["doc_id"], doc_metadata)

        # Restaurer l'état
        st.session_state.selected_doc = restored_doc
        st.session_state.document_loaded = True
        st.session_state.app_state = "chat"
        st.session_state.chat_messages = conv["messages"]
        st.session_state.pdf_bytes = pdf_bytes
        st.session_state.last_sources = []
        st.session_state.last_answer = None
        st.session_state.active_conversation_id = conv["doc_id"]

    except Exception:
        pass  # Silently fail - the PDF might no longer be available


def load_document_for_chat(doc: HALDocument):
    """Charger un document pour le chat RAG."""
    translations = get_translations()

    # Sauvegarder la conversation en cours avant d'en ouvrir une nouvelle
    _save_current_conversation()

    try:
        with st.spinner(translations.get("loading_title")):
            st.info(translations.get("loading_downloading"))
            pdf_bytes = st.session_state.hal_client.download_pdf(doc.pdf_url, doc.doc_id)

            if st.session_state.rag_engine is None:
                st.info(translations.get("loading_initializing"))
                st.session_state.rag_engine = RAGEngine()

            st.info(translations.get("loading_processing"))
            doc_metadata = {"title": doc.title, "author": doc.author, "doc_id": doc.doc_id}

            ingestion_stats = st.session_state.rag_engine.ingest_document(pdf_bytes, doc.doc_id, doc_metadata)

            st.session_state.selected_doc = doc
            st.session_state.document_loaded = True
            st.session_state.app_state = "chat"
            st.session_state.chat_messages = []
            st.session_state.pdf_bytes = pdf_bytes
            st.session_state.active_conversation_id = doc.doc_id

            st.success(
                translations.get("loading_success", chunks=ingestion_stats['total_chunks'], chars=ingestion_stats['total_characters'])
            )
            st.rerun()

    except Exception as e:
        st.error(translations.get("loading_error", error=str(e)))


def display_pdf(pdf_bytes: bytes, height: int = 800, sources: List[Dict[str, Any]] = None, answer_text: str = None):
    """Afficher le PDF avec surlignage des sources et des termes de la réponse."""

    # Décider quel PDF afficher
    pdf_to_display = pdf_bytes
    is_highlighted = False

    # Si des sources sont disponibles, générer le PDF avec surlignages
    if sources and len(sources) > 0:
        try:
            with st.spinner("🎨 Génération du surlignage..."):
                pdf_to_display = PDFAnnotator.generate_highlighted_pdf(
                    pdf_bytes, sources, max_sources=8, answer_text=answer_text
                )
                is_highlighted = True
        except Exception as e:
            st.warning(f"⚠️ Impossible de générer les surlignages: {e}")
            pdf_to_display = pdf_bytes

    # Afficher le PDF dans l'iframe
    base64_pdf = base64.b64encode(pdf_to_display).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="{height}" style="border: none;"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

    # Afficher les informations
    if is_highlighted:
        st.success(f"✅ PDF avec surlignages • {len(sources)} source(s)")

        # Liste des pages avec sources
        pages_with_sources = sorted(set(s.get("page_num", 1) for s in sources))
        colors_html = ""
        for i, page in enumerate(pages_with_sources):
            color = PDFAnnotator.COLORS[i % len(PDFAnnotator.COLORS)]
            colors_html += f'<span style="background-color: {color}; padding: 2px 8px; margin: 2px; border-radius: 3px; display: inline-block;">📄 Page {page}</span>'

        st.markdown(f"**Sources surlignées sur les pages:** {colors_html}", unsafe_allow_html=True)

        # Bouton de téléchargement du PDF annoté
        st.download_button(
            label="📥 Télécharger PDF avec surlignages",
            data=pdf_to_display,
            file_name="document_annotated.pdf",
            mime="application/pdf",
            help="Télécharger le PDF avec les sources surlignées"
        )
    else:
        st.caption("📄 PDF affiché (pas de sources à surligner)")


def render_chat_phase():
    """Afficher la phase de chat RAG."""
    translations = get_translations()

    col_header1, col_header2 = st.columns([4, 1])
    with col_header1:
        st.header(translations.get("chat_header"))
    with col_header2:
        if st.session_state.pdf_bytes:
            toggle_label = translations.get("chat_hide_pdf") if st.session_state.show_pdf else translations.get("chat_show_pdf")
            if st.button(toggle_label, use_container_width=True):
                st.session_state.show_pdf = not st.session_state.show_pdf
                st.rerun()

    if st.session_state.selected_doc:
        with st.expander(translations.get("chat_current_doc"), expanded=False):
            doc = st.session_state.selected_doc
            st.markdown(f"{translations.get('chat_doc_title')} {doc.title}")
            st.markdown(f"{translations.get('chat_doc_author')} {doc.author}")
            st.markdown(f"{translations.get('chat_doc_date')} {doc.publication_date or translations.get('na')}")
            st.markdown(f"[{translations.get('chat_doc_view_hal')}]({doc.url})")

    st.divider()

    if st.session_state.show_pdf and st.session_state.pdf_bytes:
        chat_col, pdf_col = st.columns([1.2, 1])
    else:
        chat_col = st.container()
        pdf_col = None

    with chat_col:
        chat_container = st.container()

        with chat_container:
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

                    if message["role"] == "assistant" and "sources" in message:
                        with st.expander(translations.get("chat_sources_header"), expanded=False):
                            for i, source in enumerate(message["sources"]):
                                # Badge de navigation coloré
                                page_num = source.get("page_num", 1)
                                color = PDFAnnotator.COLORS[i % len(PDFAnnotator.COLORS)]

                                col_src1, col_src2 = st.columns([5, 1])
                                with col_src1:
                                    st.markdown(
                                        f"""
                                        <div class="source-box">
                                        <strong>{translations.get("chat_source_chunk", num=i+1, chunk=source['chunk_index'])}</strong>
                                        <span style="background-color: {color}; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; margin-left: 8px;">📄 Page {page_num}</span><br>
                                        {source['content']}
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                                with col_src2:
                                    if st.button(f"📍 P.{page_num}", key=f"nav_hist_{message.get('msg_id', 0)}_{i}", help="Aller à cette page"):
                                        st.session_state.current_page = page_num
                                        st.session_state.last_sources = message["sources"]
                                        st.session_state.last_answer = message.get("content")
                                        st.rerun()

        if prompt := st.chat_input(translations.get("chat_input_placeholder")):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})

            with chat_container:
                with st.chat_message("user"):
                    st.write(prompt)

                with st.chat_message("assistant"):
                    with st.spinner(translations.get("chat_thinking")):
                        try:
                            response = st.session_state.rag_engine.ask_question(prompt)
                            answer = response["answer"]
                            sources = response["sources"]

                            st.write(answer)

                            with st.expander(translations.get("chat_sources_header"), expanded=True):
                                for i, source in enumerate(sources):
                                    # Badge de navigation coloré
                                    page_num = source.get("page_num", 1)
                                    color = PDFAnnotator.COLORS[i % len(PDFAnnotator.COLORS)]

                                    col_src1, col_src2 = st.columns([5, 1])
                                    with col_src1:
                                        st.markdown(
                                            f"""
                                            <div class="source-box">
                                            <strong>{translations.get("chat_source_chunk", num=i+1, chunk=source['chunk_index'])}</strong>
                                            <span style="background-color: {color}; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; margin-left: 8px;">📄 Page {page_num}</span><br>
                                            {source['content']}
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )
                                    with col_src2:
                                        if st.button(f"📍 P.{page_num}", key=f"nav_new_{i}", help="Aller à cette page"):
                                            st.session_state.current_page = page_num
                                            st.session_state.last_sources = sources
                                            st.rerun()

                            # Mettre à jour les sources et la réponse pour l'affichage PDF
                            st.session_state.last_sources = sources
                            st.session_state.last_answer = answer

                            msg_id = len(st.session_state.chat_messages)
                            st.session_state.chat_messages.append(
                                {"role": "assistant", "content": answer, "sources": sources, "msg_id": msg_id}
                            )

                        except Exception as e:
                            error_msg = translations.get("chat_error", error=str(e))
                            st.error(error_msg)
                            st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})

    if pdf_col and st.session_state.pdf_bytes:
        with pdf_col:
            st.subheader(translations.get("chat_pdf_viewer"))
            # Passer les dernières sources et la réponse pour le surlignage
            display_pdf(
                st.session_state.pdf_bytes, height=750,
                sources=st.session_state.last_sources,
                answer_text=st.session_state.last_answer
            )


def reset_application():
    """Réinitialiser l'application à l'état initial."""
    # Sauvegarder la conversation en cours avant de reset
    _save_current_conversation()

    if st.session_state.rag_engine:
        st.session_state.rag_engine.reset()

    st.session_state.app_state = "search"
    st.session_state.search_results = None
    st.session_state.selected_doc = None
    st.session_state.document_loaded = False
    st.session_state.chat_messages = []
    st.session_state.pdf_bytes = None
    st.session_state.show_pdf = True
    st.session_state.selected_domains = []
    st.session_state.selected_methodologies = []
    st.session_state.active_conversation_id = None
    st.session_state.show_summaries = {}
    st.session_state.current_page = 1
    st.session_state.current_annotations = []
    st.session_state.last_sources = []
    st.session_state.last_answer = None


def main():
    """Point d'entrée principal de l'application."""
    init_session_state()

    translations = get_translations()
    translations.set_language(st.session_state.language)

    render_header()
    render_sidebar()

    if st.session_state.app_state == "search":
        render_search_phase()
    elif st.session_state.app_state == "chat":
        render_chat_phase()


if __name__ == "__main__":
    main()
