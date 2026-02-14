# Synapse - Guide de Revision LangChain & Recherche Vectorielle

> Document de preparation pour la presentation orale.
> Objectif : savoir expliquer chaque fonction Python, le pipeline RAG, et la recherche vectorielle.

---

## 1. LA DATA : D'ou vient-elle et qu'est-ce qui est vectorise ?

### Source des donnees

On utilise l'**API HAL Science** (`https://api.archives-ouvertes.fr/search/`). C'est une API REST publique qui donne acces a toutes les theses deposees sur HAL.

**On ne stocke PAS de data en local.** Tout est dynamique :
1. L'utilisateur tape une recherche (ex: "machine learning")
2. On interroge l'API HAL en temps reel
3. L'API retourne une liste de theses (titre, auteur, resume, URL du PDF)
4. Quand l'utilisateur choisit une these, on **telecharge le PDF a la volee**
5. On **vectorise le contenu du PDF** en memoire (RAM, pas de base persistante)

### Ce qui est vectorise

| Element | Vectorise ? | Utilisation |
|---------|------------|-------------|
| **Texte du PDF** (page par page) | **OUI** | C'est la donnee principale vectorisee |
| Resume (abstract) | Non | Affiche dans l'UI + recommandations |
| Titre | Non | Metadata dans les chunks |
| Auteur | Non | Metadata dans les chunks |
| Mots-cles | Non | Metadata seulement |

**Concretement** : On extrait tout le texte du PDF avec PyMuPDF (`fitz`), on le decoupe en chunks, et ces chunks sont transformes en vecteurs (embeddings) puis indexes dans FAISS.

### Ce qu'on peut dire au prof pour la data

> "Notre data provient de l'API HAL Science en temps reel. On ne stocke rien en local. Quand l'utilisateur choisit une these, on telecharge le PDF, on extrait le texte page par page avec PyMuPDF, puis on le vectorise avec un modele d'embeddings. Le prof peut chercher n'importe quel sujet de these - par exemple 'intelligence artificielle', 'changement climatique' - et la data sera recuperee en live depuis HAL."

---

## 2. PIPELINE COMPLET DE RECHERCHE VECTORIELLE

```
Utilisateur tape "machine learning"
         |
         v
[1] HALAPIClient.search_theses("machine learning")
    -> Appel REST a api.archives-ouvertes.fr
    -> Retourne liste de HALDocument (titre, auteur, PDF URL, etc.)
         |
         v
[2] Utilisateur clique "Discuter avec ce document"
         |
         v
[3] HALAPIClient.download_pdf(pdf_url)
    -> Telecharge le PDF en bytes
         |
         v
[4] PDFProcessor.extract_text_with_pages(pdf_bytes)
    -> PyMuPDF (fitz) extrait le texte page par page
    -> Retourne [(page_1, "texte..."), (page_2, "texte..."), ...]
         |
         v
[5] CHUNKING (decoupage du texte)
    Deux methodes possibles :

    a) Hierarchique (si la structure est detectee) :
       ThesisParser detecte la table des matieres
       HierarchicalChunker cree 3 niveaux de chunks :
       - Chapitre (1500 chars) -> contexte large
       - Section (1000 chars)  -> contexte moyen
       - Paragraphe (500 chars) -> precision

    b) Plat (fallback) :
       RecursiveCharacterTextSplitter(chunk_size=1000, overlap=200)
       Decoupe avec separateurs : ["\n\n", "\n", ". ", " ", ""]
         |
         v
[6] EMBEDDING (transformation en vecteurs)
    Chaque chunk de texte -> vecteur de 384/1536 dimensions

    Deux options :
    - HuggingFace : sentence-transformers/all-MiniLM-L6-v2 (384 dim)
    - OpenAI : text-embedding-3-small (1536 dim)
         |
         v
[7] INDEXATION FAISS (stockage des vecteurs)

    Multi-Index (4 indexes paralleles) :
    - Chapter Index  -> chunks de chapitre
    - Section Index  -> chunks de section
    - Paragraph Index -> chunks de paragraphe
    - Full Index     -> TOUS les chunks

    Chaque index a son propre retriever hybride (FAISS + BM25)
         |
         v
[8] QUESTION de l'utilisateur
    "Quelle est la methodologie utilisee ?"
         |
         v
[9] QUERY ROUTER
    Analyse la question avec des regex patterns :
    - "methodologie" -> route vers Section Index
    - "page 42" -> route vers Paragraph Index
    - "chapitre 3" -> route vers Chapter Index
    - question generale -> route vers Full Index
         |
         v
[10] RETRIEVAL HYBRIDE (FAISS + BM25)

     FAISS (60%) : recherche par similarite vectorielle
       -> MMR (Maximal Marginal Relevance) pour diversite
       -> Recupere 24 candidats, garde les 8 meilleurs

     BM25 (40%) : recherche par mots-cles (TF-IDF)
       -> Matching exact de termes
       -> Complementaire a FAISS

     EnsembleRetriever combine les deux avec ponderation
         |
         v
[11] CONTEXT EXPANDER
     Enrichit les chunks recuperes avec le contexte :
     - expand_context(window=1) : ajoute chunks avant/apres
     - expand_by_section() : ajoute tout le contenu de la section
     - expand_by_chapter() : ajoute tout le chapitre
         |
         v
[12] LLM (GPT-4 Turbo)
     Prompt : "Tu es un assistant specialise... Contexte: [chunks] Question: [...]"
     -> Genere la reponse en se basant sur le contexte
         |
         v
[13] FILTRAGE DES SOURCES
     _filter_relevant_sources() :
     - Verifie que chaque source contient des mots-cles de la question
     - Score par mots-cles de la reponse
     - Garde les 5 sources les plus pertinentes
         |
         v
[14] AFFICHAGE
     - Reponse du LLM
     - Sources avec numeros de page
     - PDF surligne (PDFAnnotator)
```

---

## 3. COMPOSANTS LANGCHAIN - EXPLICATION DETAILLEE

### 3.1 `RecursiveCharacterTextSplitter` (Text Splitting)

**Fichiers** : `rag_engine.py:486`, `hierarchical_chunker.py:54-70`

**A quoi ca sert** : Decouper un long texte en morceaux (chunks) de taille fixe.

**Comment ca marche** :
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Taille max de chaque chunk (en caracteres)
    chunk_overlap=200,      # Chevauchement entre chunks (pour ne pas couper les idees)
    separators=["\n\n", "\n", ". ", " ", ""],  # Priorite de coupure
)
chunks = text_splitter.split_text("Mon long texte...")
```

**Pourquoi "Recursive"** : Il essaie de couper au premier separateur (`\n\n` = double saut de ligne). Si le chunk est encore trop grand, il essaie le suivant (`\n`), puis `. `, etc. Ca preserve la structure du texte au maximum.

**Pourquoi un overlap** : Si une phrase importante est a cheval sur 2 chunks, l'overlap (200 chars) garantit qu'elle apparait dans les deux. Sinon on pourrait perdre du contexte.

**Ce qu'on peut dire au prof** :
> "On utilise RecursiveCharacterTextSplitter de LangChain pour decouper le texte du PDF en chunks de 1000 caracteres avec 200 de chevauchement. Le 'Recursive' signifie qu'il coupe d'abord aux paragraphes, puis aux lignes, puis aux phrases - pour garder la coherence semantique."

---

### 3.2 `Document` (Schema LangChain)

**Fichier** : Utilise partout (`rag_engine.py:14`, `hierarchical_chunker.py:6`, etc.)

**A quoi ca sert** : C'est la structure de donnees standard de LangChain. Chaque chunk devient un `Document` avec :
- `page_content` : le texte du chunk
- `metadata` : dictionnaire avec les infos associees (page, chapitre, auteur, etc.)

```python
from langchain.schema import Document

doc = Document(
    page_content="Le texte du chunk...",
    metadata={
        "doc_id": "hal-123456",
        "page_num": 42,
        "chunk_index": 15,
        "level": 2,            # 1=chapitre, 2=section, 3=paragraphe
        "chunk_type": "section",
        "chapter_title": "Chapitre 3",
        "section_title": "Methodologie",
    }
)
```

**Ce qu'on peut dire au prof** :
> "Chaque chunk est encapsule dans un objet Document de LangChain qui contient le texte et des metadonnees : numero de page, niveau hierarchique, titre du chapitre. Ces metadonnees sont cruciales pour le surlignage PDF et la navigation dans les sources."

---

### 3.3 `FAISS` (Vector Store)

**Fichier** : `rag_engine.py:8`, `multi_index_rag.py:6`

**A quoi ca sert** : Stocker les embeddings (vecteurs) et faire des recherches par similarite.

**FAISS = Facebook AI Similarity Search**. C'est une librairie de Meta optimisee pour chercher les vecteurs les plus proches dans un grand ensemble.

```python
from langchain_community.vectorstores import FAISS

# Creation de l'index : transforme les Documents en vecteurs et les stocke
vector_store = FAISS.from_documents(documents, embeddings)

# Recherche : trouve les chunks les plus proches de la question
retriever = vector_store.as_retriever(
    search_type="mmr",           # Maximal Marginal Relevance
    search_kwargs={
        "k": 8,                  # Nombre de resultats
        "fetch_k": 24,           # Candidats evalues (3x k)
        "lambda_mult": 0.7,      # 70% pertinence, 30% diversite
    }
)
```

**MMR (Maximal Marginal Relevance)** : Au lieu de prendre les 8 chunks les plus similaires (qui pourraient tous dire la meme chose), MMR equilibre pertinence et diversite. `lambda_mult=0.7` = 70% pertinence + 30% diversite.

**Batch creation** : Pour les gros documents, on cree FAISS par batches de 100 docs pour eviter les limites de tokens OpenAI.

**Ce qu'on peut dire au prof** :
> "On utilise FAISS de Meta comme vector store. Quand on vectorise un PDF, chaque chunk est transforme en vecteur de 384 dimensions, et FAISS indexe ces vecteurs pour des recherches ultra-rapides. On utilise MMR (Maximal Marginal Relevance) au lieu d'une simple similarite cosinus, ce qui garantit des resultats a la fois pertinents ET divers - on ne veut pas 8 chunks qui disent la meme chose."

---

### 3.4 `BM25Retriever` (Keyword Search)

**Fichier** : `rag_engine.py:9`, `multi_index_rag.py:7`

**A quoi ca sert** : Recherche par mots-cles, complementaire a FAISS.

**BM25 = Best Matching 25**. C'est un algorithme classique de recherche par mots-cles (evolution de TF-IDF). Il cherche les chunks qui contiennent les memes mots que la question.

```python
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 8  # Nombre de resultats
```

**Pourquoi BM25 en plus de FAISS** : FAISS cherche par sens semantique (un vecteur proche), BM25 cherche par mots exacts. Exemple :
- Question : "Universite de Rennes"
- FAISS pourrait retourner des chunks sur d'autres universites (semantiquement proches)
- BM25 retourne les chunks qui contiennent exactement "Rennes"

**Ce qu'on peut dire au prof** :
> "On combine FAISS (recherche semantique) avec BM25 (recherche par mots-cles) dans un systeme hybride. FAISS comprend le sens, BM25 matche les termes exacts. C'est crucial pour les noms propres : si on cherche 'Universite de Rennes', FAISS pourrait retourner des chunks sur d'autres universites car semantiquement proches, mais BM25 trouve les occurrences exactes de 'Rennes'."

---

### 3.5 `EnsembleRetriever` (Hybrid Search)

**Fichier** : `rag_engine.py:10`, `multi_index_rag.py:8`

**A quoi ca sert** : Combiner FAISS et BM25 avec des poids.

```python
from langchain.retrievers import EnsembleRetriever

hybrid_retriever = EnsembleRetriever(
    retrievers=[faiss_retriever, bm25_retriever],
    weights=[0.6, 0.4],  # 60% FAISS + 40% BM25
)
```

**Comment ca marche** :
1. FAISS retourne ses 8 meilleurs chunks (avec scores de similarite)
2. BM25 retourne ses 8 meilleurs chunks (avec scores BM25)
3. EnsembleRetriever fusionne les deux listes en ponderant : 60% poids FAISS, 40% poids BM25
4. Retourne les 8 meilleurs de la fusion

**Ce qu'on peut dire au prof** :
> "L'EnsembleRetriever fusionne les resultats de FAISS et BM25 avec une ponderation 60/40. C'est une approche hybrid search qui prend le meilleur des deux mondes : la comprehension semantique de FAISS et la precision lexicale de BM25."

---

### 3.6 `OpenAIEmbeddings` (Embeddings)

**Fichier** : `rag_engine.py:11`

**A quoi ca sert** : Transformer du texte en vecteurs numeriques.

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # Modele OpenAI
)

# Embed un texte -> vecteur de 1536 dimensions
vector = embeddings.embed_query("Mon texte")  # -> [0.023, -0.451, 0.187, ...]
```

**Alternative HuggingFace** : On a aussi un wrapper custom pour sentence-transformers :
```python
class HuggingFaceEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):      # Pour les chunks
        return self.model.encode(texts).tolist()

    def embed_query(self, text):           # Pour la question
        return self.model.encode([text])[0].tolist()
```

**Ce qu'on peut dire au prof** :
> "On utilise OpenAI text-embedding-3-small pour les embeddings, ou en alternative sentence-transformers de HuggingFace. Chaque chunk de texte est transforme en vecteur de 1536 dimensions (OpenAI) ou 384 (HuggingFace). On a cree un wrapper custom pour HuggingFace qui imite l'interface de LangChain avec `embed_documents()` et `embed_query()`."

---

### 3.7 `ChatOpenAI` (LLM)

**Fichier** : `rag_engine.py:11`

**A quoi ca sert** : C'est le modele de langage qui genere les reponses.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model_name="gpt-4-turbo-preview",
    temperature=0.7,  # Creativite (0=deterministe, 1=creatif)
)

# Utilisation directe
response = llm.predict("Quelle est la capitale de la France ?")
```

**Ce qu'on peut dire au prof** :
> "On utilise GPT-4 Turbo via ChatOpenAI de LangChain. Il recoit le contexte (les chunks recuperes) et la question, puis genere une reponse. La temperature est a 0.7 pour un bon equilibre entre precision et fluidite."

---

### 3.8 `ConversationBufferMemory` (Memoire)

**Fichier** : `rag_engine.py:13`

**A quoi ca sert** : Garder l'historique de conversation pour que le LLM se souvienne des echanges precedents.

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",    # Cle dans le prompt
    return_messages=True,          # Format messages (pas string)
    output_key="answer",           # Quelle sortie stocker
)
```

**Comment ca marche** : A chaque question/reponse, la memoire stocke l'echange. Quand l'utilisateur pose une nouvelle question, tout l'historique est envoye au LLM comme contexte. Ca permet des questions de suivi : "Et qu'en est-il de la partie 2 ?" -> le LLM sait de quoi on parle.

**Ce qu'on peut dire au prof** :
> "ConversationBufferMemory stocke tout l'historique de la conversation pour que le chatbot puisse repondre a des questions de suivi. Si je demande 'quel est l'auteur ?' puis 'et son laboratoire ?', le LLM sait que 'son' fait reference a l'auteur mentionne precedemment."

---

### 3.9 `ConversationalRetrievalChain` (La chaine principale)

**Fichier** : `rag_engine.py:12`

**A quoi ca sert** : C'est LE composant central. Il orchestre tout le pipeline RAG.

```python
from langchain.chains import ConversationalRetrievalChain

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,                          # GPT-4 Turbo
    retriever=hybrid_retriever,        # EnsembleRetriever (FAISS + BM25)
    memory=memory,                     # ConversationBufferMemory
    return_source_documents=True,      # Retourner les sources
    verbose=False,
)

# Utilisation
result = conversation_chain({"question": "Quelle est la methodologie ?"})
answer = result["answer"]              # Reponse du LLM
sources = result["source_documents"]   # Documents sources utilises
```

**Ce que fait la chaine en interne** :
1. Prend la question + l'historique de chat
2. Reformule la question si necessaire (en tenant compte du contexte)
3. Appelle le retriever (FAISS + BM25) pour trouver les chunks pertinents
4. Construit un prompt avec les chunks comme contexte
5. Envoie au LLM (GPT-4)
6. Retourne la reponse + les sources

**Ce qu'on peut dire au prof** :
> "ConversationalRetrievalChain est le coeur de notre RAG. Elle prend la question, utilise le retriever hybride pour trouver les chunks pertinents dans le PDF vectorise, les injecte comme contexte dans le prompt du LLM, et genere une reponse avec les sources. Elle gere aussi l'historique de conversation grace a la memoire."

---

## 4. ARCHITECTURE AVANCEE (3 Phases)

### Phase 1 : Hybrid RAG (FAISS + BM25)
- **Quoi** : Combinaison recherche vectorielle + mots-cles
- **Pourquoi** : Les vecteurs seuls ratent les noms propres et termes techniques
- **Config** : 60% FAISS (semantique) + 40% BM25 (lexical)

### Phase 2 : Chunking Hierarchique + Multi-Index
- **Quoi** : 3 niveaux de chunks (chapitre/section/paragraphe) + 4 indexes FAISS
- **Pourquoi** : Une question sur la "methodologie" n'a pas besoin du meme niveau de detail qu'une question sur "la definition de X a la page 42"
- **Comment** : ThesisParser extrait la structure (TOC du PDF), HierarchicalChunker cree les chunks, MultiIndexRAG gere 4 indexes paralleles

### Phase 3 : Query Router + Context Expander
- **QueryRouter** : Analyse la question et la dirige vers le bon index
  - "methodologie" -> Section Index
  - "page 42" -> Paragraph Index
  - "resume general" -> Chapter Index
  - question vague -> Full Index
- **ContextExpander** : Ajoute les chunks voisins pour plus de contexte
  - `expand_context(window=1)` : +-1 chunk autour
  - `expand_by_section()` : tous les chunks de la meme section

---

## 5. FONCTIONS CLE A CONNAITRE (par fichier)

### `src/api_client.py`

| Fonction | Role |
|----------|------|
| `HALAPIClient.search_theses(query)` | Appelle l'API HAL, retourne une liste de `HALDocument` |
| `HALAPIClient.download_pdf(pdf_url)` | Telecharge le PDF en bytes |
| `HALAPIClient._parse_document(doc)` | Convertit la reponse JSON de HAL en objet `HALDocument` |

### `src/rag_engine.py`

| Fonction | Role |
|----------|------|
| `PDFProcessor.extract_text_with_pages(pdf_bytes)` | Extrait le texte page par page avec PyMuPDF |
| `RAGEngine.__init__()` | Initialise embeddings (HuggingFace ou OpenAI) + LLM |
| `RAGEngine.ingest_document(pdf_bytes, doc_id, metadata)` | **Pipeline principal** : extraction -> chunking -> embedding -> FAISS -> retriever |
| `RAGEngine._create_flat_chunks(pages_text, ...)` | Chunking plat avec RecursiveCharacterTextSplitter |
| `RAGEngine._create_faiss_batched(documents, embeddings)` | Cree FAISS par batch de 100 docs |
| `RAGEngine.ask_question(question)` | Pose une question via ConversationalRetrievalChain |
| `RAGEngine._ask_question_advanced(question)` | Version avancee : QueryRouter -> MultiIndex -> ContextExpander -> LLM |
| `RAGEngine._filter_relevant_sources(answer, sources, question)` | Filtre les sources par pertinence (mots-cles question + reponse) |
| `RAGEngine._keyword_search_documents(query)` | Recherche par mots-cles brute dans tous les documents |
| `HuggingFaceEmbeddings` | Wrapper custom pour rendre SentenceTransformer compatible LangChain |

### `src/hierarchical_chunker.py`

| Fonction | Role |
|----------|------|
| `HierarchicalChunker.create_hierarchical_chunks(sections)` | Cree les chunks a 3 niveaux depuis les sections |
| `HierarchicalChunker.convert_to_documents(chunks)` | Convertit les chunks en Documents LangChain |

### `src/multi_index_rag.py`

| Fonction | Role |
|----------|------|
| `MultiIndexRAG.build_indexes(chunks, documents)` | Construit les 4 indexes FAISS (chapter/section/paragraph/full) |
| `MultiIndexRAG._create_hybrid_retriever(faiss, docs, name)` | Cree un EnsembleRetriever pour un index donne |
| `MultiIndexRAG.get_retriever(index_type)` | Retourne le retriever de l'index demande |

### `src/query_router.py`

| Fonction | Role |
|----------|------|
| `QueryRouter.route_query(query)` | Route la question vers le bon index par regex |
| `QueryRouter.route_query_with_llm(query)` | Route en utilisant le LLM (plus intelligent, plus lent) |

### `src/context_expander.py`

| Fonction | Role |
|----------|------|
| `ContextExpander.expand_context(docs, window)` | Ajoute les chunks voisins (+-window) |
| `ContextExpander.expand_by_chapter(docs)` | Ajoute tous les chunks du meme chapitre |
| `ContextExpander.expand_by_section(docs)` | Ajoute tous les chunks de la meme section |

---

## 6. QUESTIONS FREQUENTES DU PROF (et reponses)

### "C'est quoi un embedding ?"
> Un embedding c'est une representation numerique d'un texte sous forme de vecteur. Par exemple, "chat" -> [0.2, -0.5, 0.8, ...] avec 384 dimensions. Deux textes semantiquement proches auront des vecteurs proches (mesure par similarite cosinus).

### "Pourquoi FAISS et pas Chroma ou Pinecone ?"
> FAISS (Meta) est la reference pour la recherche de similarite vectorielle. C'est extremement rapide, open source, et fonctionne en memoire - ideal pour notre cas ou on vectorise un seul PDF a la fois. Pas besoin d'un serveur externe.

### "Pourquoi hybrid search (FAISS + BM25) ?"
> FAISS seul comprend le sens mais peut rater les termes exacts. Exemple : chercher "Universite de Rennes", FAISS pourrait retourner des chunks sur n'importe quelle universite. BM25 matche les mots exacts. Ensemble, on couvre les deux cas.

### "C'est quoi MMR ?"
> Maximal Marginal Relevance. Au lieu de retourner les 8 chunks les plus similaires (qui pourraient etre redondants), MMR penalise les resultats trop similaires entre eux. Ca garantit des resultats pertinents ET divers. Notre lambda est a 0.7 : 70% pertinence, 30% diversite.

### "Pourquoi 3 niveaux de chunks ?"
> Une question sur la "conclusion generale" a besoin d'un contexte large (chunk de 1500 chars = chapitre). Une question sur "la definition de X" a besoin d'un chunk precis (500 chars = paragraphe). Le chunking hierarchique adapte la granularite au type de question.

### "Comment le chatbot sait vers quel index chercher ?"
> Le QueryRouter analyse la question avec des patterns regex. Par exemple, s'il detecte "methodologie" ou "resultats", il route vers le Section Index. S'il detecte "page 42", il route vers le Paragraph Index. Par defaut, il utilise le Full Index qui cherche partout.

### "C'est quoi le scraping ici ?"
> Techniquement, on ne fait pas de scraping classique (pas de BeautifulSoup/Selenium). On utilise l'API REST officielle de HAL Science - c'est une API publique avec des endpoints JSON. Pour le PDF, on le telecharge directement via son URL. L'extraction de texte se fait avec PyMuPDF (fitz) qui parse le format PDF.

### "Qu'est-ce qui se passe si l'API HAL est down ?"
> On a un timeout de 30 secondes sur l'API et 120 secondes pour le telechargement PDF. Si ca echoue, un message d'erreur est affiche. Mais HAL est une infrastructure academique stable, rarement indisponible.

### "Comment vous gerez les gros PDF ?"
> Limite de 50 MB. Le texte est extrait page par page (pas tout en memoire d'un coup). Les embeddings sont crees par batch de 100 documents pour eviter les limites de tokens de l'API OpenAI. FAISS est aussi cree par batch.

---

## 7. CHEAT SHEET : IMPORTS LANGCHAIN DU PROJET

```python
# Text Splitting - Decoupage du texte en chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector Store - Stockage et recherche de vecteurs
from langchain_community.vectorstores import FAISS

# Retrievers - Recuperation de documents
from langchain_community.retrievers import BM25Retriever    # Mots-cles
from langchain.retrievers import EnsembleRetriever           # Fusion hybride

# Embeddings & LLM - Modeles d'IA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Chaine de conversation RAG
from langchain.chains import ConversationalRetrievalChain

# Memoire conversationnelle
from langchain.memory import ConversationBufferMemory

# Structure de donnees
from langchain.schema import Document
```

---

## 8. SCHEMA VISUEL SIMPLIFIE (pour le pitch)

```
  [Utilisateur]
       |
   "machine learning"
       |
       v
  [API HAL Science]  -----> Liste de theses
       |
  [Telecharge PDF]
       |
       v
  [PyMuPDF]  -----> Texte brut (page par page)
       |
       v
  [RecursiveCharacterTextSplitter]  -----> Chunks de texte
       |
       v
  [Embeddings]  -----> Vecteurs numeriques
  (OpenAI ou HuggingFace)
       |
       v
  [FAISS]  -----> Index vectoriel en memoire
  (4 indexes : chapter/section/paragraph/full)
       |
       v
  [Question utilisateur]
       |
       v
  [QueryRouter]  -----> Choisit le bon index
       |
       v
  [EnsembleRetriever]  -----> 8 chunks pertinents
  (60% FAISS + 40% BM25)
       |
       v
  [ContextExpander]  -----> Enrichit avec contexte
       |
       v
  [GPT-4 Turbo]  -----> Reponse + Sources
       |
       v
  [PDFAnnotator]  -----> PDF surligne
```
