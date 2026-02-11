# Chatbot RAG Sciences HAL 🔬

Un chatbot spécialisé avec génération augmentée par récupération (RAG) pour la recherche scientifique utilisant les [archives ouvertes HAL Science](https://hal.science). Recherchez des thèses scientifiques et engagez des conversations intelligentes avec le contenu des documents.

## Fonctionnalités

- **🔍 Phase de Découverte** : Recherche et filtrage de thèses dans les archives HAL Science
- **💬 Phase d'Approfondissement** : Discutez avec les documents sélectionnés via un pipeline RAG avancé
- **📄 Traitement PDF Intelligent** : Extraction automatique de texte, découpage et vectorisation
- **🎯 Citations de Sources** : Visualisez les chunks de documents pertinents supportant chaque réponse
- **⚡ Performance Optimisée** : Base vectorielle FAISS pour une recherche de similarité rapide
- **🌐 Multilingue** : Interface disponible en français et anglais
- **🔧 Configurable** : Support pour les embeddings OpenAI ou HuggingFace

## Architecture

```
API HAL → Téléchargement PDF → Extraction Texte → Découpage → Embeddings → Base Vectorielle → Chat LLM
```

### Stack Technique

- **Frontend** : Streamlit
- **Framework RAG** : LangChain
- **Base Vectorielle** : FAISS
- **Traitement PDF** : PyMuPDF (fitz)
- **Embeddings** : OpenAI ou HuggingFace (sentence-transformers)
- **LLM** : OpenAI GPT-4

## Installation

### 1. Cloner et Configurer

```bash
cd hal-rag-chatbot
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configurer l'Environnement

Copiez le fichier d'environnement exemple et ajoutez vos clés API :

```bash
cp .env.example .env
```

Éditez `.env` et ajoutez votre clé API OpenAI :

```env
OPENAI_API_KEY=sk-votre-cle-ici
```

**Note** : Une clé API OpenAI est requise pour la fonctionnalité complète (embeddings + LLM). Alternativement, vous pouvez utiliser les embeddings gratuits HuggingFace pour la base vectorielle, mais vous devrez configurer un LLM local séparément.

### 3. Lancer l'Application

```bash
streamlit run app.py
```

L'application s'ouvrira dans votre navigateur à `http://localhost:8501`

## Utilisation

### Phase 1 : Recherche et Découverte

1. Entrez des mots-clés de recherche (ex : "apprentissage automatique", "changement climatique")
2. Parcourez les résultats avec titres, résumés, auteurs
3. Sélectionnez une thèse avec PDF disponible pour continuer

### Phase 2 : Discussion avec le Document

1. Attendez le traitement du PDF (découpage et vectorisation automatiques)
2. Posez des questions sur le contenu du document
3. Recevez des réponses générées par IA avec citations de sources
4. Visualisez les chunks de documents pertinents supportant chaque réponse

## Changement de Langue

L'application démarre en **français** par défaut. Pour changer la langue :

1. Ouvrez la sidebar (barre latérale)
2. En haut, vous verrez "🌐 Langue"
3. Sélectionnez "English" ou "Français" dans le menu déroulant
4. L'interface se met à jour automatiquement

## Options de Configuration

Éditez le fichier `.env` pour personnaliser :

| Variable | Description | Défaut |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Clé API OpenAI | Requis |
| `OPENAI_MODEL` | Modèle GPT pour le chat | `gpt-4-turbo-preview` |
| `CHUNK_SIZE` | Taille des chunks de texte (caractères) | `1000` |
| `CHUNK_OVERLAP` | Chevauchement entre chunks | `200` |
| `TOP_K_RESULTS` | Nombre de chunks pertinents à récupérer | `4` |
| `USE_HUGGINGFACE_EMBEDDINGS` | Utiliser HuggingFace gratuit au lieu d'OpenAI | `False` |
| `VECTOR_STORE_TYPE` | Backend de base vectorielle | `faiss` |

## Structure du Projet

```
hal-rag-chatbot/
├── app.py                    # Application Streamlit principale
├── src/
│   ├── api_client.py         # Wrapper API HAL
│   ├── rag_engine.py         # Pipeline RAG (PDF→Base Vectorielle→Chat)
│   ├── config.py             # Gestion de configuration
│   ├── translations.py       # Système multilingue (i18n)
│   └── utils.py              # Fonctions utilitaires
├── requirements.txt          # Dépendances Python
├── .env.example              # Modèle de variables d'environnement
├── README.md                 # Documentation (anglais)
└── README_FR.md              # Documentation (français)
```

## Composants Clés

### Client API HAL (`src/api_client.py`)

- Recherche dans les archives HAL avec filtrage pour les thèses
- Gestion de la pagination et récupération d'erreurs
- Téléchargement de PDFs depuis les serveurs HAL
- Parsing des métadonnées (titre, auteur, résumé, mots-clés)

### Moteur RAG (`src/rag_engine.py`)

- **Traitement PDF** : Extraction et nettoyage de texte depuis les PDFs
- **Découpage de Texte** : RecursiveCharacterTextSplitter avec chevauchement
- **Vectorisation** : Création d'embeddings via OpenAI ou HuggingFace
- **Base Vectorielle** : FAISS pour une recherche de similarité efficace
- **Chaîne Conversationnelle** : LangChain ConversationalRetrievalChain
- **Mémoire** : Maintien de l'historique de chat et du contexte

### Système de Traductions (`src/translations.py`)

- Support multilingue (français/anglais)
- Sélecteur de langue dynamique
- Toutes les chaînes de l'interface sont traduites
- Facilement extensible à d'autres langues

## Dépannage

### Pas de Clé API OpenAI

Si vous n'avez pas de clé API OpenAI :
1. Définissez `USE_HUGGINGFACE_EMBEDDINGS=True` dans `.env`
2. Cela active les embeddings gratuits mais nécessite une configuration LLM supplémentaire
3. Considérez l'utilisation d'Ollama ou d'autres LLMs locaux

### Échec du Téléchargement de PDF

- Certains PDFs peuvent être restreints ou derrière authentification
- Vérifiez l'URL HAL directement dans un navigateur
- Essayez une autre thèse avec PDF disponible

### Problèmes de Mémoire

Pour les PDFs volumineux :
- Réduisez `CHUNK_SIZE` dans `.env`
- Réduisez `TOP_K_RESULTS` pour récupérer moins de chunks
- Augmentez la RAM système disponible

### Performance Lente

- Utilisez `gpt-3.5-turbo` au lieu de `gpt-4` pour des réponses plus rapides
- Réduisez le nombre de chunks (`TOP_K_RESULTS`)
- Utilisez FAISS au lieu de ChromaDB pour la base vectorielle

## Feuille de Route

- [ ] Support pour la comparaison de plusieurs documents
- [ ] Export de l'historique de chat en PDF/Markdown
- [ ] Filtres avancés (plage de dates, domaine, institution)
- [ ] Génération de citations dans divers formats
- [ ] Support multi-langues étendu
- [ ] Intégration avec d'autres archives scientifiques (arXiv, PubMed)

## Contribuer

Les contributions sont bienvenues ! Domaines d'amélioration :
- Meilleur parsing de PDF pour les mises en page complexes
- Support pour bases vectorielles supplémentaires (Pinecone, Weaviate)
- UI/UX améliorée dans Streamlit
- Tests unitaires et d'intégration
- Conteneurisation Docker

## Licence

Licence MIT - libre d'utilisation pour la recherche et l'éducation.

## Remerciements

- [HAL Science](https://hal.science) pour fournir un accès ouvert à la recherche scientifique
- [LangChain](https://langchain.com) pour le framework RAG
- [OpenAI](https://openai.com) pour les embeddings et LLM
- [Streamlit](https://streamlit.io) pour le framework web

---

**Bonne recherche !** 🔬📚
