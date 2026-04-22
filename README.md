# 🅿️ Système Intelligent de Contrôle de Parking

Ce projet est une solution complète de détection et de reconnaissance automatique de plaques d'immatriculation (ALPR) pour la gestion d'un parking. Il utilise l'intelligence artificielle pour identifier les véhicules et vérifier leur autorisation d'accès en temps réel.

## 🏗️ Architecture du Système

Le système est divisé en plusieurs microservices orchestrés par **Docker Compose** :

1. **Le Backend (FastAPI) :** API RESTful hautement performante traitant les flux d'images en mémoire vive.
2. **Le Moteur d'IA :** - *Vision :* Modèle YOLOv8 entraîné sur mesure pour la détection des plaques.
   - *OCR :* EasyOCR optimisé avec prétraitement d'image (Filtre Bilatéral, Algorithme d'Otsu).
3. **Le MLOps (MLflow & SQLite) :** Base de données assurant la traçabilité de chaque inférence, des métriques de confiance et de l'état des véhicules.
4. **Le Frontend (Streamlit) :** Interface tablette réactive permettant l'analyse d'images statiques, de vidéos et le streaming réseau en temps réel (IP Camera / Phone Link).

## 🚀 Fonctionnalités

- **Détection de Plaques :** Utilisation de **YOLOv8** pour localiser précisément les plaques d'immatriculation.
- **OCR (Reconnaissance de Caractères) :** Utilisation de **EasyOCR** pour lire le texte des plaques.
- **Vérification d'Autorisation :** Comparaison automatique avec une base de données d'abonnés.
- **Multi-Sources :** Supporte l'importation d'images, de vidéos, de flux caméra en direct et de caméras IP.
- **MLOps :** Suivi des inférences et des performances avec **MLflow**.
- **Interface Intuitive :** Interface utilisateur moderne construite avec **Streamlit**.
- **Architecture Robuste :** Backend performant avec **FastAPI**.

## 🛠️ Technologies Utilisées

- **Vision par Ordinateur :** OpenCV, numPy, Pillow, Ultralytics YOLOv8
- **OCR :** EasyOCR
- **Backend :** FastAPI, Pydantic, Uvicorn, Requests
- **Frontend :** Streamlit
- **Tracking & MLOps :** MLflow
- **Containerisation et DevOps :** Docker, Docker Compose, SQLite

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

Assurez-vous d'avoir **Docker** et **Docker Compose** installés sur votre machine.

1. Clonez ce dépôt :

   ```bash
   git clone [https://github.com/MichelAI123/Parking_Detection.git](https://github.com/MichelAI123/Parking_Detection.git)
   cd Parking_Detection
2. Lancez l'application :

   ```bash
   docker-compose up --build
   ```

3. Accédez à l'interface :
   - Frontend : `http://localhost:8501`
   - API Docs : `http://localhost:8000/docs`

### Installation Manuelle

Assurez-vous d'avoir **Docker** et **Docker Compose** installés sur votre machine.

1. Clonez ce dépôt :

   ```bash
   git clone [https://github.com/MichelAI123/Parking_Detection.git](https://github.com/MichelAI123/Parking_Detection.git)
   cd Parking_Detection 
   
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
   python -m uvicorn main:app --reload 
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

## 📝 Auteurs

- **MichelAI123** - [GitHub](https://github.com/MichelAI123)
- **Arole KENFACK** - [GitHub](https://github.com/kenarole)

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.
