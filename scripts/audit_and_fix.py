"""Comprehensive audit and bug detection script."""

import sys
from pathlib import Path
import importlib

print("🔍 AUDIT COMPLET DU SYSTÈME")
print("=" * 60)

bugs_found = []
warnings = []

# Test 1: Vérifier les imports
print("\n1️⃣ TEST DES IMPORTS")
print("-" * 60)

sys.path.insert(0, str(Path.cwd()))

try:
    from src.config import config
    print("✅ src.config importé")

    print(f"   ENABLE_CLASSIFICATION: {config.ENABLE_CLASSIFICATION}")
    print(f"   ENABLE_SUMMARIZATION: {config.ENABLE_SUMMARIZATION}")
    print(f"   ENABLE_RECOMMENDATIONS: {config.ENABLE_RECOMMENDATIONS}")

except Exception as e:
    bugs_found.append(f"❌ ERREUR config.py: {e}")
    print(bugs_found[-1])

try:
    from src.translations import get_translations
    trans = get_translations()
    print("✅ src.translations importé")
    print(f"   Langue actuelle: {trans.language}")
except Exception as e:
    bugs_found.append(f"❌ ERREUR translations.py: {e}")
    print(bugs_found[-1])

try:
    from src.api_client import HALAPIClient
    client = HALAPIClient()
    print("✅ src.api_client importé")
except Exception as e:
    bugs_found.append(f"❌ ERREUR api_client.py: {e}")
    print(bugs_found[-1])

try:
    from src.rag_engine import RAGEngine
    print("✅ src.rag_engine importé")
except Exception as e:
    bugs_found.append(f"❌ ERREUR rag_engine.py: {e}")
    print(bugs_found[-1])

# Test ML imports (optionnel)
print("\n2️⃣ TEST DES MODULES ML")
print("-" * 60)

if 'config' in dir() and config.ENABLE_CLASSIFICATION:
    try:
        from src.ml_features.classifier import ThesisClassifier
        classifier = ThesisClassifier(use_ml_model=False)
        print("✅ Classifier importé et initialisé")
    except Exception as e:
        bugs_found.append(f"❌ ERREUR classifier.py: {e}")
        print(bugs_found[-1])

if 'config' in dir() and config.ENABLE_SUMMARIZATION:
    try:
        from src.ml_features.summarizer import ThesisSummarizer
        summarizer = ThesisSummarizer(use_transformers=False)
        print("✅ Summarizer importé et initialisé")
    except Exception as e:
        bugs_found.append(f"❌ ERREUR summarizer.py: {e}")
        print(bugs_found[-1])

if 'config' in dir() and config.ENABLE_RECOMMENDATIONS:
    try:
        from src.ml_features.recommender import ThesisRecommender
        recommender = ThesisRecommender()
        print("✅ Recommender importé et initialisé")
    except Exception as e:
        bugs_found.append(f"❌ ERREUR recommender.py: {e}")
        print(bugs_found[-1])

# Test 3: Vérifier la syntaxe de app.py
print("\n3️⃣ TEST SYNTAXE DE APP.PY")
print("-" * 60)

try:
    import ast
    with open("app.py", "r") as f:
        code = f.read()
    ast.parse(code)
    print("✅ app.py: Syntaxe valide")
except SyntaxError as e:
    bugs_found.append(f"❌ ERREUR SYNTAXE app.py ligne {e.lineno}: {e.msg}")
    print(bugs_found[-1])
except Exception as e:
    bugs_found.append(f"❌ ERREUR app.py: {e}")
    print(bugs_found[-1])

# Test 4: Vérifier les dépendances
print("\n4️⃣ TEST DES DÉPENDANCES")
print("-" * 60)

required_packages = {
    "streamlit": "Interface web",
    "langchain": "RAG framework",
    "openai": "LLM API",
    "sentence_transformers": "Embeddings",
    "faiss": "Vector store",
    "requests": "HTTP client",
    "fitz": "PDF processing (PyMuPDF)",
}

for package, description in required_packages.items():
    try:
        if package == "fitz":
            import fitz
        else:
            importlib.import_module(package)
        print(f"✅ {package:25} - {description}")
    except ImportError:
        warnings.append(f"⚠️  {package:25} - NON INSTALLÉ")
        print(warnings[-1])

# Test 5: Vérifier .env
print("\n5️⃣ TEST CONFIGURATION .ENV")
print("-" * 60)

env_file = Path(".env")
if env_file.exists():
    print("✅ .env existe")

    # Vérifier la clé OpenAI
    if 'config' in dir() and config.OPENAI_API_KEY:
        print(f"✅ OPENAI_API_KEY configurée ({config.OPENAI_API_KEY[:10]}...)")
    else:
        warnings.append("⚠️  OPENAI_API_KEY non configurée")
        print(warnings[-1])
else:
    bugs_found.append("❌ .env n'existe pas")
    print(bugs_found[-1])

# Test 6: Vérifier la structure des fichiers
print("\n6️⃣ TEST STRUCTURE DES FICHIERS")
print("-" * 60)

required_files = [
    "app.py",
    "src/config.py",
    "src/api_client.py",
    "src/rag_engine.py",
    "src/translations.py",
    "src/utils.py",
    "src/ml_features/__init__.py",
    "src/ml_features/classifier.py",
    "src/ml_features/summarizer.py",
    "src/ml_features/recommender.py",
]

for file_path in required_files:
    if Path(file_path).exists():
        print(f"✅ {file_path}")
    else:
        bugs_found.append(f"❌ FICHIER MANQUANT: {file_path}")
        print(bugs_found[-1])

# Test 7: Test fonctionnel rapide
print("\n7️⃣ TEST FONCTIONNEL")
print("-" * 60)

try:
    # Test classification
    if 'config' in dir() and 'classifier' in dir() and config.ENABLE_CLASSIFICATION:
        text = "Machine learning and artificial intelligence"
        result = classifier.classify(text)
        print(f"✅ Classification fonctionne: {result['domains'][:2]}")

    # Test summarization
    if 'config' in dir() and 'summarizer' in dir() and config.ENABLE_SUMMARIZATION:
        text = "This is a test. " * 50
        summary = summarizer.generate_summaries(text)
        print(f"✅ Summarization fonctionne: {len(summary['tldr'])} chars")

    # Test recommender
    if 'config' in dir() and 'recommender' in dir() and config.ENABLE_RECOMMENDATIONS:
        recommender.index_thesis("test1", "AI and ML", {"title": "Test"})
        stats = recommender.get_statistics()
        print(f"✅ Recommender fonctionne: {stats['total_theses_indexed']} thèses")

except Exception as e:
    bugs_found.append(f"❌ ERREUR test fonctionnel: {e}")
    print(bugs_found[-1])

# Résumé
print("\n" + "=" * 60)
print("📊 RÉSUMÉ DE L'AUDIT")
print("=" * 60)

print(f"\n🐛 Bugs critiques trouvés: {len(bugs_found)}")
for bug in bugs_found:
    print(f"   {bug}")

print(f"\n⚠️  Avertissements: {len(warnings)}")
for warning in warnings:
    print(f"   {warning}")

if len(bugs_found) == 0 and len(warnings) == 0:
    print("\n✅ AUCUN BUG DÉTECTÉ! Le système est opérationnel.")
elif len(bugs_found) == 0:
    print("\n✅ Aucun bug critique. Quelques avertissements mineurs.")
else:
    print("\n❌ Des bugs critiques nécessitent une correction.")

# Sauvegarder le rapport
with open("AUDIT_REPORT.txt", "w") as f:
    f.write("RAPPORT D'AUDIT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Bugs critiques: {len(bugs_found)}\n")
    for bug in bugs_found:
        f.write(f"  {bug}\n")
    f.write(f"\nAvertissements: {len(warnings)}\n")
    for warning in warnings:
        f.write(f"  {warning}\n")

print("\n📄 Rapport sauvegardé: AUDIT_REPORT.txt")
