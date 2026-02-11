"""Script to activate ML features in the app."""

import re
from pathlib import Path

print("🚀 Activation des fonctionnalités ML...")
print("=" * 60)

app_path = Path("app.py")
app_content = app_path.read_text()

# Modification 1: Ajouter les imports ML
print("\n1️⃣ Ajout des imports ML...")

imports_to_add = """
# ML Features
if config.ENABLE_CLASSIFICATION:
    from src.ml_features.classifier import ThesisClassifier
if config.ENABLE_SUMMARIZATION:
    from src.ml_features.summarizer import ThesisSummarizer
if config.ENABLE_RECOMMENDATIONS:
    from src.ml_features.recommender import ThesisRecommender
"""

# Trouver la ligne après les imports existants
import_pattern = r"(from src\.translations import get_translations, t)"
app_content = re.sub(
    import_pattern,
    r"\1" + imports_to_add,
    app_content,
    count=1
)

print("   ✓ Imports ML ajoutés")

# Modification 2: Ajouter l'initialisation ML dans init_session_state
print("\n2️⃣ Ajout de l'initialisation ML...")

ml_init = """
    # ML Features initialization
    if config.ENABLE_CLASSIFICATION and "ml_classifier" not in st.session_state:
        st.session_state.ml_classifier = ThesisClassifier(use_ml_model=config.USE_ML_MODELS)
    if config.ENABLE_SUMMARIZATION and "ml_summarizer" not in st.session_state:
        st.session_state.ml_summarizer = ThesisSummarizer(use_transformers=config.USE_ML_MODELS)
    if config.ENABLE_RECOMMENDATIONS and "ml_recommender" not in st.session_state:
        st.session_state.ml_recommender = ThesisRecommender()

    # ML Filters state
    if "selected_domains" not in st.session_state:
        st.session_state.selected_domains = []
    if "selected_methodologies" not in st.session_state:
        st.session_state.selected_methodologies = []
"""

init_pattern = r"(if \"show_pdf\" not in st\.session_state:\s+st\.session_state\.show_pdf = True.*?)((?:\n\n|\s*def ))"
app_content = re.sub(
    init_pattern,
    r"\1" + ml_init + r"\2",
    app_content,
    count=1,
    flags=re.DOTALL
)

print("   ✓ Initialisation ML ajoutée")

# Modifier 3: Ajouter les filtres ML dans la sidebar
print("\n3️⃣ Ajout des filtres ML dans la sidebar...")

filters_code = """
        # ML Smart Filters
        if config.ENABLE_CLASSIFICATION and st.session_state.get('search_results'):
            st.divider()
            st.subheader(translations.get("ml_filters_header"))

            # Extract unique domains and methodologies from results
            all_domains = set()
            all_methodologies = set()

            for doc in st.session_state.search_results.get("docs", []):
                if hasattr(doc, 'classification'):
                    all_domains.update(doc.classification.get('domains', []))
                    all_methodologies.update(doc.classification.get('methodologies', []))

            if all_domains:
                selected_domains = st.multiselect(
                    translations.get("ml_domains"),
                    sorted(list(all_domains)),
                    default=st.session_state.selected_domains
                )
                st.session_state.selected_domains = selected_domains

            if all_methodologies:
                selected_methods = st.multiselect(
                    translations.get("ml_methodologies"),
                    sorted(list(all_methodologies)),
                    default=st.session_state.selected_methodologies
                )
                st.session_state.selected_methodologies = selected_methods
"""

# Ajouter avant le About dans render_sidebar
sidebar_pattern = r"(# About\s+st\.divider\(\))"
app_content = re.sub(
    sidebar_pattern,
    filters_code + r"\n        \1",
    app_content,
    count=1
)

print("   ✓ Filtres ML ajoutés dans la sidebar")

# Écrire le fichier modifié
print("\n4️⃣ Sauvegarde du fichier modifié...")

# Backup de l'original
backup_path = Path("app_original_backup.py")
if not backup_path.exists():
    backup_path.write_text(app_path.read_text())
    print(f"   ✓ Backup créé: {backup_path}")

app_path.write_text(app_content)
print(f"   ✓ {app_path} mis à jour")

print("\n" + "=" * 60)
print("✅ ACTIVATION ML TERMINÉE!")
print("=" * 60)
print("\n📋 Prochaines étapes:")
print("   1. Copier .env.example vers .env (si pas déjà fait)")
print("   2. Lancer: streamlit run app.py")
print("   3. Tester les nouvelles fonctionnalités!")
print("\n💡 Les fonctionnalités sont activées via .env:")
print("   ENABLE_CLASSIFICATION=True")
print("   ENABLE_SUMMARIZATION=True")
print("   ENABLE_RECOMMENDATIONS=True")
