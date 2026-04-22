import re
import os
import csv
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import easyocr
import mlflow
import sqlite3
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
        self.base_dir = Path(os.getcwd())
        self.db_path = self.base_dir / "mlflow.db"
        self.tracking_uri = f"sqlite:///{self.db_path.as_posix()}"
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(ALPRConfig.EXP_NAME)

    def log_inference(self, plaque: str, confiance: float, statut: str):
        try:
            with mlflow.start_run(run_name="Requete_API"):
                mlflow.log_param("plaque", plaque)
                mlflow.log_param("statut", statut)
                mlflow.log_metric("confiance", confiance)
        except Exception:
            pass

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
        self.base_dir = Path(os.getcwd())
        model_path = self.base_dir / model_filename
        self.model = YOLO(str(model_path) if model_path.exists() else "yolov8n.pt")
        self.ocr_service = OCRService()
        self.mlops = MLOpsService()
        
        # Initialisation du Fichier Log CSV
        self.log_file = self.base_dir / "journal_analyses.csv"
        if not self.log_file.exists():
            with open(self.log_file, mode='w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(["Date_Heure", "Plaque", "Statut", "Confiance"])
        
        # Initialisation Base de donnees principale SQLite
        self.parking_db_path = self.base_dir / "parking.db"
        self.initialiser_base_donnees()

    def _get_db_connection(self):
        conn = sqlite3.connect(self.parking_db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialiser_base_donnees(self):
        try:
            conn = self._get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS vehicules_autorises (
                    plaque TEXT PRIMARY KEY,
                    date_ajout DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Liste des plaques fournies par l'administration (nettoyée des doublons)
            plaques_fournies = [
                "CVLR185", "DEAK951", "6475255965", "DHJB730", "FIA794", "FAAR794", 
                "OHJB815", "IDHJB815", "JDHJB815", "DDYF855", "DOYF855", "DYF855", 
                "DEJC312", "DEJC3126", "97TTS", "197I5", "127IT9", "DITP36F", "0TP366", 
                "ITP366", "1813", "I89EPG", "JY971", "CSH7JG", "CSH719", "DCSM719", 
                "DHHY339", "HHY339", "I91PSB"
            ]
            
            # Injection de masse (INSERT OR IGNORE evite les erreurs si la plaque existe deja)
            for p in set(plaques_fournies):
                cur.execute("INSERT OR IGNORE INTO vehicules_autorises (plaque) VALUES (?);", (p,))
                
            conn.commit()
            conn.close()
            print("[INFO] Base SQLite (parking.db) initialisee avec les plaques par defaut.")
        except Exception as e:
            print(f"[ERREUR] Connexion SQLite: {e}")

    # --- METHODES DE GESTION DE BASE DE DONNEES (CRUD) ---
    def est_autorise(self, plaque: str) -> bool:
        try:
            conn = self._get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM vehicules_autorises WHERE plaque = ?;", (plaque,))
            autorise = cur.fetchone() is not None
            conn.close()
            return autorise
        except:
            return False

    def lister_plaques(self):
        try:
            conn = self._get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT plaque, date_ajout FROM vehicules_autorises ORDER BY date_ajout DESC;")
            resultats = [{"plaque": row["plaque"], "date_ajout": row["date_ajout"]} for row in cur.fetchall()]
            conn.close()
            return resultats
        except:
            return []

    def ajouter_plaque(self, plaque: str):
        try:
            conn = self._get_db_connection()
            cur = conn.cursor()
            cur.execute("INSERT OR IGNORE INTO vehicules_autorises (plaque) VALUES (?);", (plaque,))
            conn.commit()
            conn.close()
            return True
        except:
            return False

    def supprimer_plaque(self, plaque: str):
        try:
            conn = self._get_db_connection()
            cur = conn.cursor()
            cur.execute("DELETE FROM vehicules_autorises WHERE plaque = ?;", (plaque,))
            conn.commit()
            conn.close()
            return True
        except:
            return False

    # --- MOTEUR D'INFERENCE ---
    def traiter_image_memoire(self, img_array: np.ndarray):
        hauteur_img, largeur_img = img_array.shape[:2]
        results = self.model.predict(source=img_array, conf=ALPRConfig.CONF_MIN_YOLO, verbose=False)
        resultats_requete = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                marge = ALPRConfig.MARGE_CROP_PIXELS
                y1_crop, y2_crop = max(0, y1 - marge), min(hauteur_img, y2 + marge)
                x1_crop, x2_crop = max(0, x1 - marge), min(largeur_img, x2 + marge)
                
                plaque_crop = img_array[y1_crop:y2_crop, x1_crop:x2_crop]
                if plaque_crop.size == 0: continue
                
                meilleure_plaque = self.ocr_service.lire_plaque(plaque_crop)
                
                if meilleure_plaque:
                    texte = meilleure_plaque["texte"]
                    confiance = meilleure_plaque["confiance"]
                    statut = "AUTORISE" if self.est_autorise(texte) else "NON AUTORISE"
                    
                    self.mlops.log_inference(texte, confiance, statut)
                    
                    try:
                        with open(self.log_file, mode='a', newline='', encoding='utf-8') as f:
                            csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), texte, statut, f"{confiance:.2f}"])
                    except Exception as e:
                        print(f"Erreur ecriture log: {e}")

                    resultats_requete.append({
                        "plaque": texte, "confiance": confiance, "statut": statut,
                        "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                    })
        return resultats_requete

app = FastAPI(title="API Controle Parking La Cite")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

systeme_alpr = None

@app.on_event("startup")
async def demarrage_serveur():
    global systeme_alpr
    systeme_alpr = APIPipeline(model_filename="best.pt")

@app.post("/api/v1/analyze/image")
async def analyser_image_upload(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    resultats = systeme_alpr.traiter_image_memoire(img)
    return {"success": len(resultats) > 0, "data": resultats}

@app.get("/api/v1/database/plates")
def obtenir_plaques():
    return {"success": True, "data": systeme_alpr.lister_plaques()}

@app.post("/api/v1/database/plates/{plaque}")
def ajouter_plaque(plaque: str):
    plaque_propre = re.sub(r'[^A-Z0-9]', '', plaque.upper())
    succes = systeme_alpr.ajouter_plaque(plaque_propre)
    return {"success": succes, "message": "Ajout effectue" if succes else "Erreur d'ajout"}

@app.delete("/api/v1/database/plates/{plaque}")
def supprimer_plaque(plaque: str):
    plaque_propre = re.sub(r'[^A-Z0-9]', '', plaque.upper())
    succes = systeme_alpr.supprimer_plaque(plaque_propre)
    return {"success": succes, "message": "Suppression effectuee" if succes else "Erreur de suppression"}