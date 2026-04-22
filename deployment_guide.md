# Guide de Déploiement

Ce document explique comment déployer l'application sur un serveur de production.

## Option 1 : Déploiement avec Docker (Recommandé)

Le projet inclut un fichier `docker-compose.yml` qui configure automatiquement le Backend (API) et le Frontend (Streamlit).

### Étapes :
1. Clonez le repository sur votre serveur.
2. Assurez-vous que Docker et Docker Compose sont installés.
3. Construisez et lancez les conteneurs :
   ```bash
   docker-compose up --build -d
   ```
4. L'application sera accessible sur le port 8501 (Frontend) et 8000 (API).

## Option 2 : Déploiement Cloud (Azure/AWS/GCP)

### Azure Container Instances (ACI)
Vous pouvez déployer les deux images (API et App) sur ACI en utilisant le contexte Docker Azure.

### Google Cloud Run
Déployez le backend sur Cloud Run et le frontend séparément, en configurant la variable d'environnement `API_URL` du frontend pour pointer vers l'URL du backend Cloud Run.

## Configuration du Frontend

Lors du déploiement, vous devez configurer l'URL de l'API pour que Streamlit puisse communiquer avec elle.
Dans `app.py`, l'URL est récupérée via une variable d'environnement :
```python
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1/analyze/image")
```

Si vous utilisez Docker Compose, cela est déjà configuré dans le fichier.
