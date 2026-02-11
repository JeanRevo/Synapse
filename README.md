# 🔬 HAL Science RAG Chatbot

Un chatbot intelligent pour la recherche scientifique utilisant les archives ouvertes HAL Science avec RAG (Retrieval-Augmented Generation) et fonctionnalités de Machine Learning.

---

## 📋 Table des Matières

- [Fonctionnalités](#-fonctionnalités)
- [Structure du Projet](#-structure-du-projet)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Configuration](#%EF%B8%8F-configuration)
- [Documentation](#-documentation)
- [Technologies](#-technologies)

---

## ✨ Fonctionnalités

### 🔍 Recherche et Découverte
- **Recherche HAL API** : Recherche dans les archives ouvertes de thèses françaises
- **Filtres intelligents ML** : Classification automatique par domaine et méthodologie
- **Affichage enrichi** : Métadonnées complètes avec badges de classification

### 💬 Chat RAG avec PDF
- **Analyse PDF complète** : Extraction et vectorisation du texte
- **Questions/Réponses contextuelles** : Chat basé sur le contenu du document
- **Visualiseur PDF intégré** : Affichage côte à côte avec le chat
- **Sources citées** : Traçabilité des réponses avec références aux chunks

### 🤖 Fonctionnalités ML
- **Classification automatique** : Détection de domaines (8 catégories), méthodologies (5 types), types de contribution
- **Résumés multi-niveaux** : TL;DR, résumé exécutif, résumé détaillé
- **Recommandations** : Thèses similaires basées sur embeddings sémantiques

### 🌍 Interface Bilingue
- Support Français / Anglais
- Traduction dynamique de l'interface
- Commentaires de code en français

---

## 📁 Structure du Projet

```
Projet IA/
├── app.py                          # Application Streamlit principale
├── requirements.txt                # Dépendances Python
├── .env                           # Variables d'environnement (non versionné)
├── .env.example                   # Template de configuration
│
├── src/                           # Code source
│   ├── __init__.py
│   ├── config.py                  # Configuration de l'application
│   ├── api_client.py              # Client API HAL
│   ├── rag_engine.py              # Moteur RAG (standard)
│   ├── rag_engine_enhanced.py     # Moteur RAG amélioré (MMR, reranking)
│   ├── translations.py            # Gestion des traductions FR/EN
│   ├── utils.py                   # Fonctions utilitaires
│   │
│   └── ml_features/               # Modules de Machine Learning
│       ├── __init__.py
│       ├── classifier.py          # Classification de thèses
│       ├── summarizer.py          # Génération de résumés
│       └── recommender.py         # Système de recommandations
│
├── docs/                          # Documentation
│   ├── README_FR.md               # Documentation en français
│   ├── INSTALLATION_COMPLETE.md   # Guide d'installation
│   ├── ML_FEATURES_GUIDE.md       # Guide des fonctionnalités ML
│   ├── RAG_AUDIT_IMPROVEMENTS.md  # Audit et améliorations RAG
│   └── AUDIT_REPORT.txt           # Dernier rapport d'audit
│
├── scripts/                       # Scripts utilitaires
│   ├── test_ml_features.py        # Tests des modules ML
│   ├── audit_and_fix.py           # Script d'audit système
│   ├── activate_ml_features.py    # Activation des fonctionnalités ML
│   └── complete_ui_integration.py # Intégration UI complète
│
└── archives/                      # Fichiers de backup
    ├── app_original_backup.py
    └── app_before_full_ui.py
```

---

## 🚀 Installation

### 1. Prérequis
- Python 3.9+
- pip
- 2GB RAM minimum (4GB recommandé pour les modèles ML)

### 2. Cloner le projet
```bash
cd "/Users/jean-lisek/Desktop/Projet IA"
```

### 3. Créer un environnement virtuel
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# ou
venv\Scripts\activate     # Windows
```

### 4. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 5. Configurer les variables d'environnement
```bash
cp .env.example .env
# Éditer .env et ajouter votre clé OpenAI
```

---

## 💻 Utilisation

### Lancer l'application
```bash
source venv/bin/activate
streamlit run app.py
```

### Workflow d'utilisation
1. **Phase 1 - Recherche** :
   - Entrer une requête (ex: "machine learning")
   - Filtrer par domaines/méthodologies (sidebar)
   - Voir les résultats avec classifications ML
   - Générer des résumés si besoin

2. **Phase 2 - Chat avec PDF** :
   - Cliquer sur "💬 Discuter" pour un document
   - Le PDF est téléchargé et analysé
   - Poser des questions sur le contenu
   - Voir les recommandations de thèses similaires

---

## ⚙️ Configuration

### Fichier `.env`

```bash
# OpenAI (Requis pour le chat RAG)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# RAG
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=4
VECTOR_STORE_TYPE=faiss

# Fonctionnalités ML
ENABLE_CLASSIFICATION=True
ENABLE_SUMMARIZATION=True
ENABLE_RECOMMENDATIONS=True
USE_ML_MODELS=False  # True pour utiliser Transformers (plus lent mais meilleur)

# Application
DEBUG=False
```

### Modes ML
- **Rule-based** (`USE_ML_MODELS=False`) : Rapide, basé sur mots-clés
- **Transformer-based** (`USE_ML_MODELS=True`) : Meilleure qualité, plus lent

---

## 📚 Documentation

- **[Installation Complète](docs/INSTALLATION_COMPLETE.md)** : Guide détaillé d'installation
- **[Guide ML](docs/ML_FEATURES_GUIDE.md)** : Documentation des fonctionnalités ML
- **[Audit RAG](docs/RAG_AUDIT_IMPROVEMENTS.md)** : Améliorations techniques du RAG
- **[README Français](docs/README_FR.md)** : Documentation originale en français

---

## 🛠 Technologies

### Backend
- **LangChain** : Orchestration RAG
- **OpenAI** : Embeddings & LLM
- **FAISS** : Base de données vectorielle
- **PyMuPDF (fitz)** : Traitement PDF
- **Sentence Transformers** : Embeddings HuggingFace

### Frontend
- **Streamlit** : Interface web
- **Pandas** : Manipulation de données

### ML
- **Transformers** : Classification zero-shot (optionnel)
- **scikit-learn** : Cosine similarity
- **torch** : Backend ML

---

## 🧪 Tests

### Tests unitaires ML
```bash
python scripts/test_ml_features.py
```

### Audit système complet
```bash
python scripts/audit_and_fix.py
```

**Dernier audit** : ✅ 0 bugs, 0 avertissements

---

## 📊 Performance

| Fonctionnalité | Temps de chargement | Mémoire |
|---|---|---|
| **Initialisation app** | 2-3s | 300MB |
| **Classification (rule-based)** | <0.1s/thèse | +50MB |
| **Classification (ML)** | 0.5s/thèse | +500MB |
| **Résumé (extractive)** | 0.2s | +100MB |
| **Recommandations** | 0.1s/recherche | +200MB |
| **Traitement PDF** | 5-10s (50 pages) | Variable |

---

## 🔧 Scripts Utiles

- `scripts/test_ml_features.py` : Valider tous les modules ML
- `scripts/audit_and_fix.py` : Vérifier l'intégrité du système
- `scripts/activate_ml_features.py` : Activer les fonctionnalités ML
- `scripts/complete_ui_integration.py` : Intégration UI complète

---

## 📝 Licence

Projet académique - HAL Science Open Access

---

## 🤝 Contribution

1. Fork le projet
2. Créer une branche (`git checkout -b feature/nouvelle-fonctionnalité`)
3. Commit (`git commit -m 'Ajout nouvelle fonctionnalité'`)
4. Push (`git push origin feature/nouvelle-fonctionnalité`)
5. Ouvrir une Pull Request

---

## 🎓 Crédits

- **HAL Science** : [https://hal.science](https://hal.science)
- **LangChain** : Framework RAG
- **OpenAI** : Modèles de langage
- **Streamlit** : Framework d'interface

---

## 📧 Support

Pour toute question ou problème :
1. Consulter la [documentation](docs/)
2. Lancer un audit : `python scripts/audit_and_fix.py`
3. Vérifier les logs d'erreur
4. Ouvrir une issue GitHub

---

**Version** : 2.0
**Dernière mise à jour** : 2026-02-11
**Status** : ✅ Production Ready
