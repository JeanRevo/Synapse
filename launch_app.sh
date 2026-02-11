#!/bin/bash

# Script de lancement sécurisé du HAL RAG Chatbot

PROJECT_DIR="/Users/jean-lisek/Desktop/Projet IA"
cd "$PROJECT_DIR"

# Couleurs pour l'affichage
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "🔬 HAL Science RAG Chatbot"
echo "================================"

# Vérifier que le venv existe
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Environnement virtuel non trouvé!${NC}"
    echo "Créez-le avec: python3 -m venv venv"
    exit 1
fi

# Activer le venv
echo -e "${YELLOW}📦 Activation de l'environnement virtuel...${NC}"
source venv/bin/activate

# Vérifier les dépendances critiques
echo -e "${YELLOW}🔍 Vérification des dépendances...${NC}"
python3 -c "import streamlit, langchain, openai, sentence_transformers" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Dépendances manquantes!${NC}"
    echo "Installez-les avec: pip install -r requirements.txt"
    exit 1
fi

# Vérifier le fichier .env
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  Fichier .env non trouvé${NC}"
    echo "Créez-le à partir de .env.example"
fi

# Lancer l'application
echo -e "${GREEN}✅ Tout est prêt!${NC}"
echo -e "${GREEN}🚀 Lancement de l'application...${NC}"
echo ""

./venv/bin/streamlit run app.py
