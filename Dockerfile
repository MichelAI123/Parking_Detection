# 1. Utiliser une image Python légère mais compatible avec OpenCV
FROM python:3.11-slim

# 2. Installer les dépendances système nécessaires pour OpenCV et EasyOCR
# Ces bibliothèques sont indispensables pour le traitement d'images sous Linux
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Définir le dossier de travail dans le conteneur
WORKDIR /app

# 4. Copier d'abord le fichier des dépendances pour optimiser le cache Docker
COPY requirements.txt .

# 5. Installer les bibliothèques Python
# --no-cache-dir réduit la taille finale de l'image
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copier tout le reste du projet (codes, modèles .pt, dossiers vidéos)
COPY . .

# 7. Créer les dossiers nécessaires au cas où ils manqueraient
RUN mkdir -p mlruns videos

# 8. Exposer les ports pour FastAPI (8000) et Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# 9. Commande de démarrage
# Note : Pour un projet Capstone, on peut démarrer l'API par défaut.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]