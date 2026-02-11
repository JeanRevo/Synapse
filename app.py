"""
HAL RAG Chatbot - Application Streamlit avec fonctionnalités ML

Un chatbot spécialisé pour la recherche scientifique utilisant les archives ouvertes HAL Science.
"""

import streamlit as st
import pandas as pd
from typing import Optional
import sys
from pathlib import Path
import base64

# Ajouter src au chemin
sys.path.insert(0, str(Path(__file__).parent))

from src.api_client import HALAPIClient, HALDocument
from src.rag_engine import RAGEngine
from src.utils import truncate_text, clean_query, validate_pdf_url
from src.config import config
from src.translations import get_translations, t

# Fonctionnalités ML
if config.ENABLE_CLASSIFICATION:
    from src.ml_features.classifier import ThesisClassifier
if config.ENABLE_SUMMARIZATION:
    from src.ml_features.summarizer import ThesisSummarizer
if config.ENABLE_RECOMMENDATIONS:
    from src.ml_features.recommender import ThesisRecommender

# Configuration de la page
st.set_page_config(
    page_title="HAL Science RAG Chatbot",
    page_icon="🔬",
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
    """Afficher la barre latérale avec configuration et statut."""
    translations = get_translations()

    with st.sidebar:
        # Sélecteur de langue
        st.header(translations.get("sidebar_language"))
        available_languages = translations.get_available_languages()
        language_options = list(available_languages.keys())
        current_index = language_options.index(st.session_state.language)

        selected_language = st.selectbox(
            label="Select language",
            options=language_options,
            format_func=lambda x: available_languages[x],
            index=current_index,
            label_visibility="collapsed",
        )

        if selected_language != st.session_state.language:
            st.session_state.language = selected_language
            translations.set_language(selected_language)
            st.rerun()

        st.divider()
        st.header(translations.get("sidebar_config"))

        # Statut de l'API
        st.subheader(translations.get("sidebar_api_status"))
        if config.OPENAI_API_KEY:
            st.success(translations.get("sidebar_openai_configured"))
        else:
            st.warning(translations.get("sidebar_openai_missing"))
            st.info(translations.get("sidebar_openai_info"))

        # État actuel
        st.subheader(translations.get("sidebar_current_state"))
        state_emoji = "🔍" if st.session_state.app_state == "search" else "💬"
        state_name = translations.get("sidebar_discovery_phase") if st.session_state.app_state == "search" else translations.get("sidebar_chat_phase")
        st.info(f"{state_emoji} {state_name}")

        # Statut du document
        if st.session_state.document_loaded:
            st.success(f"{translations.get('sidebar_doc_loaded')} {st.session_state.selected_doc.title[:50]}...")

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

            # Obtenir les recommandations
            if hasattr(st.session_state.selected_doc, 'doc_id'):
                try:
                    similar = st.session_state.ml_recommender.recommend_by_text(
                        st.session_state.selected_doc.abstract,
                        top_k=3
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

        # Bouton de réinitialisation
        st.divider()
        if st.button(translations.get("sidebar_new_search"), use_container_width=True):
            reset_application()
            st.rerun()

        # Paramètres
        st.divider()
        st.subheader(translations.get("sidebar_settings"))
        st.caption(f"{translations.get('sidebar_chunk_size')} {config.CHUNK_SIZE}")
        st.caption(f"{translations.get('sidebar_chunk_overlap')} {config.CHUNK_OVERLAP}")
        st.caption(f"{translations.get('sidebar_top_k')} {config.TOP_K_RESULTS}")

        # À propos
        st.divider()
        st.caption(translations.get("sidebar_about"))
        st.caption(f"{translations.get('sidebar_data_source')} [HAL Science](https://hal.science)")


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
                            {"title": doc.title, "author": doc.author}
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


def load_document_for_chat(doc: HALDocument):
    """Charger un document pour le chat RAG."""
    translations = get_translations()
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

            st.success(
                translations.get("loading_success", chunks=ingestion_stats['total_chunks'], chars=ingestion_stats['total_characters'])
            )
            st.rerun()

    except Exception as e:
        st.error(translations.get("loading_error", error=str(e)))


def display_pdf(pdf_bytes: bytes, height: int = 800):
    """Afficher le PDF dans un iframe en utilisant l'encodage base64."""
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="{height}" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


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
                                st.markdown(
                                    f"""
                                    <div class="source-box">
                                    <strong>{translations.get("chat_source_chunk", num=i+1, chunk=source['chunk_index'])}</strong><br>
                                    {source['content']}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

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

                            with st.expander(translations.get("chat_sources_header"), expanded=False):
                                for i, source in enumerate(sources):
                                    st.markdown(
                                        f"""
                                        <div class="source-box">
                                        <strong>{translations.get("chat_source_chunk", num=i+1, chunk=source['chunk_index'])}</strong><br>
                                        {source['content']}
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                            st.session_state.chat_messages.append(
                                {"role": "assistant", "content": answer, "sources": sources}
                            )

                        except Exception as e:
                            error_msg = translations.get("chat_error", error=str(e))
                            st.error(error_msg)
                            st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})

    if pdf_col and st.session_state.pdf_bytes:
        with pdf_col:
            st.subheader(translations.get("chat_pdf_viewer"))
            display_pdf(st.session_state.pdf_bytes, height=750)


def reset_application():
    """Réinitialiser l'application à l'état initial."""
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
    st.session_state.show_summaries = {}


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
