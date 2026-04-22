import re
from pathlib import Path
import cv2
import numpy as np
import easyocr
import mlflow
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# --- 1. CONFIGURATION ---
class ALPRConfig:
    MOTS_IGNORER = {"QUEBEC", "ONTARIO", "CANADA", "YOURSTODISCOVER", "JEMESOUVIENS"}
    CONF_MIN_YOLO = 0.6
    CONF_MIN_OCR = 0.20
    LONGUEUR_MIN_PLAQUE = 4
    MARGE_CROP_PIXELS = 5
    EXP_NAME = "Parking_Control_API"

# --- 2. SERVICES METIERS ---
class MLOpsService:
    def __init__(self):
        try:
            self.base_dir = Path(__file__).resolve().parent
        except NameError:
            self.base_dir = Path.cwd()
            
        self.db_path = self.base_dir / "mlflow.db"
        self.tracking_uri = f"sqlite:///{self.db_path.as_posix()}"
        
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(ALPRConfig.EXP_NAME)

    def log_inference(self, plaque: str, confiance: float, statut: str):
        try:
            with mlflow.start_run(run_name="Requete_API"):
                mlflow.log_param("systeme", "FastAPI_Backend")
                mlflow.log_param("plaque", plaque)
                mlflow.log_param("statut", statut)
                mlflow.log_metric("confiance", confiance)
        except Exception as e:
            print(f"[ALERTE] Echec de la journalisation MLflow : {e}")

class OCRService:
    def __init__(self, langs=['en']):
        self.reader = easyocr.Reader(langs)

    def pretraiter_image(self, img_crop):
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def lire_plaque(self, image_crop):
        image_prete = self.pretraiter_image(image_crop)
        resultats = self.reader.readtext(image_prete)
        
        candidats = []
        for (_, text, prob) in resultats:
            t_clean = re.sub(r'[^A-Z0-9]', '', text.upper())
            if len(t_clean) >= ALPRConfig.LONGUEUR_MIN_PLAQUE and any(c.isdigit() for c in t_clean):
                if t_clean not in ALPRConfig.MOTS_IGNORER and prob > ALPRConfig.CONF_MIN_OCR:
                    candidats.append({"texte": t_clean, "confiance": float(prob)})
        
        if candidats:
            candidats.sort(key=lambda x: x["confiance"], reverse=True)
            return candidats[0]
        return None

class APIPipeline:
    def __init__(self, model_filename="best.pt"):
        try:
            self.base_dir = Path(__file__).resolve().parent
        except NameError:
            self.base_dir = Path.cwd()
            
        model_path = self.base_dir / model_filename
        self.model = YOLO(str(model_path) if model_path.exists() else "yolov8n.pt")
        self.ocr_service = OCRService()
        self.mlops = MLOpsService()
        # Simulation d'une base de donnees d'abonnes
        self.base_abonnes = {"DFHP360", "CYLR602", "DJAV098"}

    def traiter_image_memoire(self, img_array: np.ndarray):
        """Prend un tableau numpy directement de la requete reseau."""
        hauteur_img, largeur_img = img_array.shape[:2]
        results = self.model.predict(source=img_array, conf=ALPRConfig.CONF_MIN_YOLO, verbose=False)
        
        resultats_requete = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                marge = ALPRConfig.MARGE_CROP_PIXELS
                y1_crop = max(0, y1 - marge)
                y2_crop = min(hauteur_img, y2 + marge)
                x1_crop = max(0, x1 - marge)
                x2_crop = min(largeur_img, x2 + marge)
                
                plaque_crop = img_array[y1_crop:y2_crop, x1_crop:x2_crop]
                if plaque_crop.size == 0: continue
                
                meilleure_plaque = self.ocr_service.lire_plaque(plaque_crop)
                
                if meilleure_plaque:
                    texte = meilleure_plaque["texte"]
                    confiance = meilleure_plaque["confiance"]
                    statut = "AUTORISE" if texte in self.base_abonnes else "NON AUTORISE"
                    
                    self.mlops.log_inference(texte, confiance, statut)
                    
                    resultats_requete.append({
                        "plaque": texte,
                        "confiance": confiance,
                        "statut": statut,
                        "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                    })

        return resultats_requete

# --- 3. INITIALISATION FASTAPI ---
app = FastAPI(
    title="API - Systeme Intelligent de Controle de Parking au College La Cite",
    description="API MLOps pour la detection et lecture de plaques d'immatriculation.",
    version="1.0.0"
)

# Configuration CORS pour autoriser l'application Streamlit a communiquer avec l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, specifier l'IP de Streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variable globale pour stocker l'orchestrateur
systeme_alpr = None

@app.on_event("startup")
async def demarrage_serveur():
    """Charge les modeles lourds dans la RAM au demarrage du serveur."""
    global systeme_alpr
    print("[INFO] Chargement des modeles d'Intelligence Artificielle...")
    systeme_alpr = APIPipeline(model_filename="best.pt")
    print("[INFO] Serveur API pret a recevoir des requetes.")

# --- 4. ENDPOINTS (ROUTES API) ---

@app.get("/")
def route_racine():
    return {"message": "API de Controle de Parking actif. Visitez /docs pour la documentation."}

@app.get("/api/health")
def verification_sante():
    """Permet au Frontend de verifier si le Backend est en ligne."""
    status = "pret" if systeme_alpr is not None else "chargement"
    return {"status": status, "modele_yolo": "actif", "modele_ocr": "actif"}

@app.post("/api/v1/analyze/image")
async def analyser_image_upload(file: UploadFile = File(...)):
    """
    Recoit une image depuis l'interface client, l'analyse en RAM, 
    et retourne un JSON structure.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier fourni n'est pas une image.")

    try:
        # 1. Lecture des octets recus par le reseau
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        
        # 2. Decodage en image OpenCV
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Erreur de decodage de l'image.")

        # 3. Execution de l'inference
        resultats = systeme_alpr.traiter_image_memoire(img)
        
        # 4. Reponse JSON
        if not resultats:
            return {"success": False, "message": "Aucune plaque lisible detectee.", "data": []}
            
        return {
            "success": True,
            "message": f"{len(resultats)} plaque(s) traitee(s).",
            "data": resultats
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))