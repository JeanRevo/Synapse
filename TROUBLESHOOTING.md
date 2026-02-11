# 🔧 Guide de Dépannage

Guide rapide pour résoudre les problèmes courants du HAL RAG Chatbot.

---

## 🚨 Problèmes Courants

### 1. Erreur d'import `transformers` / `tokenizers`

**Symptôme**:
```
ImportError: tokenizers>=0.14,<0.19 is required for a normal functioning of this module,
but found tokenizers==0.22.2
```

**Cause**: Conflit de versions entre `transformers` et `tokenizers`

**Solution**:
```bash
cd "/Users/jean-lisek/Desktop/Projet IA"
source venv/bin/activate
pip uninstall -y transformers tokenizers
pip install transformers==4.36.2 tokenizers==0.15.2
```

**Vérification**:
```bash
python3 -c "import transformers; print(f'✅ transformers {transformers.__version__}')"
```

---

### 2. Application ne démarre pas

**Symptôme**: `streamlit run app.py` ne fonctionne pas

**Solutions**:

#### A. Vérifier le venv
```bash
cd "/Users/jean-lisek/Desktop/Projet IA"
ls -la venv/  # Doit exister
source venv/bin/activate
which python3  # Doit pointer vers venv/bin/python3
```

#### B. Utiliser le script de lancement
```bash
./launch_app.sh
```

#### C. Réinstaller les dépendances
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

### 3. Module non trouvé (ModuleNotFoundError)

**Symptôme**:
```
ModuleNotFoundError: No module named 'src.config'
ModuleNotFoundError: No module named 'langchain'
```

**Solutions**:

#### Si module `src.*` manquant:
```bash
# Vérifier que vous êtes dans le bon dossier
pwd  # Doit afficher /Users/jean-lisek/Desktop/Projet IA
ls src/  # Doit lister les fichiers Python
```

#### Si module externe manquant:
```bash
source venv/bin/activate
pip install <nom-du-module>
# ou
pip install -r requirements.txt
```

---

### 4. Erreur OpenAI API

**Symptôme**:
```
openai.error.AuthenticationError: Incorrect API key provided
```

**Solution**:
```bash
# Vérifier .env
cat .env | grep OPENAI_API_KEY

# Si vide, éditer .env et ajouter:
OPENAI_API_KEY=sk-proj-...

# Relancer l'app
./launch_app.sh
```

---

### 5. PDF ne se charge pas

**Symptômes possibles**:
- "PDF download timed out"
- "Failed to download PDF"
- "PDF file too large"

**Solutions**:

#### A. Augmenter le timeout (dans `.env`):
```bash
PDF_DOWNLOAD_TIMEOUT=180  # 3 minutes au lieu de 2
```

#### B. Vérifier la taille maximale (dans `.env`):
```bash
MAX_PDF_SIZE_MB=100  # Augmenter si nécessaire
```

#### C. Vérifier la connexion HAL:
```bash
python3 -c "
from src.api_client import HALAPIClient
client = HALAPIClient()
results = client.search_theses('machine learning', rows=1)
print(f'✅ HAL API fonctionne: {results[\"numFound\"]} résultats')
"
```

---

### 6. Fonctionnalités ML ne s'affichent pas

**Symptôme**: Pas de filtres ML, pas de résumés, pas de recommandations

**Solution**:

#### Vérifier la configuration (`.env`):
```bash
cat .env | grep ENABLE

# Devrait afficher:
# ENABLE_CLASSIFICATION=True
# ENABLE_SUMMARIZATION=True
# ENABLE_RECOMMENDATIONS=True
```

#### Si False, modifier `.env`:
```bash
ENABLE_CLASSIFICATION=True
ENABLE_SUMMARIZATION=True
ENABLE_RECOMMENDATIONS=True
```

#### Relancer l'application:
```bash
./launch_app.sh
```

---

### 7. Erreur FAISS / Embeddings

**Symptôme**:
```
ImportError: cannot import name 'IndexFlatL2' from 'faiss'
RuntimeError: FAISS index not found
```

**Solution**:
```bash
source venv/bin/activate
pip uninstall -y faiss faiss-cpu faiss-gpu
pip install faiss-cpu==1.7.4
```

---

### 8. Mémoire insuffisante

**Symptôme**: Application lente ou crash

**Solutions**:

#### A. Désactiver les modèles ML lourds (`.env`):
```bash
USE_ML_MODELS=False  # Utilise rule-based (plus rapide, moins de RAM)
```

#### B. Réduire les chunks RAG (`.env`):
```bash
CHUNK_SIZE=500       # Au lieu de 1000
CHUNK_OVERLAP=100    # Au lieu de 200
TOP_K_RESULTS=2      # Au lieu de 4
```

#### C. Vérifier la RAM disponible:
```bash
# macOS
vm_stat | grep free

# Recommandé: Au moins 2GB libres
```

---

### 9. Avertissement urllib3/OpenSSL

**Symptôme**:
```
NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+,
currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'
```

**Impact**: ⚠️ **NON BLOQUANT** - Simple avertissement, l'application fonctionne

**Solution (optionnelle)**:
```bash
# Ignorer l'avertissement - il n'affecte pas le fonctionnement
# Ou downgrade urllib3:
pip install "urllib3<2.0"
```

---

### 10. Erreur au lancement Streamlit

**Symptôme**:
```
streamlit: command not found
```

**Solution**:
```bash
source venv/bin/activate
pip install streamlit==1.31.0
# Ou utiliser le chemin complet:
./venv/bin/streamlit run app.py
```

---

## 🧪 Tests de Diagnostic

### Test Complet du Système

```bash
cd "/Users/jean-lisek/Desktop/Projet IA"
source venv/bin/activate
python scripts/audit_and_fix.py
```

**Résultat attendu**: ✅ 0 bugs, 0 avertissements

### Test des Modules ML

```bash
python scripts/test_ml_features.py
```

**Résultat attendu**:
```
✅ Classification OK
✅ Summarization OK
✅ Recommendations OK
```

### Test des Imports

```bash
python3 -c "
from src.config import config
from src.api_client import HALAPIClient
from src.rag_engine import RAGEngine
from src.ml_features.classifier import ThesisClassifier
print('✅ Tous les imports OK')
"
```

---

## 📊 Commandes Utiles

### Vérifier les versions
```bash
source venv/bin/activate
pip list | grep -E "(streamlit|langchain|openai|transformers)"
```

### Nettoyer le cache
```bash
# Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Streamlit cache
rm -rf ~/.streamlit/cache/

# Transformers cache
rm -rf ~/.cache/huggingface/
```

### Réinstallation complète
```bash
# ATTENTION: Supprime tout et réinstalle
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🆘 Contact Support

Si le problème persiste:

1. **Vérifier les logs**:
   ```bash
   ./venv/bin/streamlit run app.py 2>&1 | tee app.log
   ```

2. **Lancer l'audit**:
   ```bash
   python scripts/audit_and_fix.py > audit_output.txt
   ```

3. **Vérifier la configuration**:
   ```bash
   cat .env
   cat src/config.py
   ```

4. **Ouvrir une issue** avec:
   - Message d'erreur complet
   - Sortie de `audit_and_fix.py`
   - Versions Python et système

---

## ✅ Checklist de Vérification Rapide

Avant de chercher plus loin, vérifiez:

- [ ] Environnement virtuel activé (`source venv/bin/activate`)
- [ ] Dépendances installées (`pip list | grep streamlit`)
- [ ] Fichier `.env` existe et contient `OPENAI_API_KEY`
- [ ] Dans le bon répertoire (`pwd` → `/Users/jean-lisek/Desktop/Projet IA`)
- [ ] Python 3.9+ (`python3 --version`)
- [ ] Suffisamment de RAM libre (≥ 2GB)
- [ ] Connexion internet active (pour HAL API)

---

**Dernière mise à jour**: 2026-02-11
**Version**: 2.0
