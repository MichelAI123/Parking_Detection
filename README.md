# 🅿️ Système Intelligent de Contrôle de Parking

Ce projet est une solution complète de détection et de reconnaissance automatique de plaques d'immatriculation (ALPR) pour la gestion d'un parking. Il utilise l'intelligence artificielle pour identifier les véhicules et vérifier leur autorisation d'accès en temps réel.

## 🚀 Fonctionnalités

- **Détection de Plaques :** Utilisation de **YOLOv8** pour localiser précisément les plaques d'immatriculation.
- **OCR (Reconnaissance de Caractères) :** Utilisation de **EasyOCR** pour lire le texte des plaques.
- **Vérification d'Autorisation :** Comparaison automatique avec une base de données d'abonnés.
- **Multi-Sources :** Supporte l'importation d'images, de vidéos, de flux caméra en direct et de caméras IP.
- **MLOps :** Suivi des inférences et des performances avec **MLflow**.
- **Interface Intuitive :** Interface utilisateur moderne construite avec **Streamlit**.
- **Architecture Robuste :** Backend performant avec **FastAPI**.

## 🛠️ Technologies Utilisées

- **Vision par Ordinateur :** OpenCV, Ultralytics YOLOv8
- **OCR :** EasyOCR
- **Backend :** FastAPI, Pydantic
- **Frontend :** Streamlit
- **Tracking & MLOps :** MLflow
- **Containerisation :** Docker, Docker Compose

## 📂 Structure du Projet

```text
├── app.py                 # Interface Frontend Streamlit
├── main.py                # Backend FastAPI (Logique ALPR)
├── requirements.txt       # Dépendances générales
├── Dockerfile             # Dockerfile pour le déploiement complet
├── docker-compose.yml     # Orchestration des services API et App
├── models/                # (Optionnel) Emplacement des modèles .pt
└── images/                # Images de test et assets
```

## ⚙️ Installation et Utilisation

### Avec Docker (Recommandé)

1. Assurez-vous d'avoir Docker et Docker Compose installés.
2. Lancez l'application :
   ```bash
   docker-compose up --build
   ```
3. Accédez à l'interface :
   - Frontend : `http://localhost:8501`
   - API Docs : `http://localhost:8000/docs`

### Installation Manuelle

1. Clonez le repository.
2. Créez un environnement virtuel :
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```
3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
4. Lancez le backend :
   ```bash
   uvicorn main:app --reload
   ```
5. Lancez le frontend :
   ```bash
   streamlit run app.py
   ```

## 📊 MLOps avec MLflow

Le système enregistre chaque détection dans une base de données locale `mlflow.db`. Pour visualiser les logs et les statistiques :
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 📝 Auteur

**MichelAI123** - [GitHub](https://github.com/MichelAI123)

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.
