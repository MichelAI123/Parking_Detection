import streamlit as st
import requests
import cv2
import numpy as np
import os
import tempfile
import pandas as pd
from datetime import datetime
from collections import deque
from PIL import Image

# --- 1. CONFIGURATION DYNAMIQUE ---
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1/analyze/image")
API_DB_URL = os.getenv("API_DB_URL", "http://127.0.0.1:8000/api/v1/database/plates")

st.set_page_config(page_title="Controle Parking - La Cite", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    [data-testid="stImage"] img { max-height: 65vh !important; object-fit: contain !important; margin: 0 auto !important; display: block !important; }
    [data-testid="stCameraInput"] video { max-height: 60vh !important; }
    </style>
""", unsafe_allow_html=True)

if "camera_active" not in st.session_state: st.session_state.camera_active = False
if "alpr_history" not in st.session_state: st.session_state.alpr_history = deque(maxlen=50)

# --- 2. INTERFACE UTILISATEUR ---
st.title("Systeme Intelligent de Controle de Parking")
st.markdown("---")

st.sidebar.header("Etat du Systeme")
st.sidebar.success("Interface operationnelle")
st.sidebar.info("Journalisation CSV active")

onglet_import_img, onglet_import_vid, onglet_camera, onglet_stream, onglet_admin = st.tabs([
    "📷 Importer Image", "🎥 Importer Video", "📱 Snapshot", "📡 Flux Continu", "⚙️ Administration DB"
])

def interroger_api(image_bytes, nom_fichier):
    try:
        files = {"file": (nom_fichier, image_bytes, "image/jpeg")}
        reponse = requests.post(API_URL, files=files, timeout=3) 
        if reponse.status_code == 200: return reponse.json()
        return None
    except requests.exceptions.RequestException: return None

def enregistrer_historique(plaque, statut, confiance):
    if len(st.session_state.alpr_history) > 0:
        derniere = st.session_state.alpr_history[0]
        if derniere["Plaque"] == plaque and derniere["Statut"] == statut:
            st.session_state.alpr_history[0]["Heure"] = datetime.now().strftime("%H:%M:%S")
            return
            
    st.session_state.alpr_history.appendleft({
        "Heure": datetime.now().strftime("%H:%M:%S"),
        "Plaque": plaque, "Statut": statut, "Confiance": f"{confiance:.2f}"
    })

def obtenir_tableau_style():
    if len(st.session_state.alpr_history) == 0:
        return pd.DataFrame(columns=["Heure", "Plaque", "Statut", "Confiance"])
    df = pd.DataFrame(st.session_state.alpr_history)
    def styliser_statut(val):
        couleur = '#2ecc71' if val == 'AUTORISE' else '#e74c3c'
        return f'color: {couleur}; font-weight: bold;'
    try: return df.style.map(styliser_statut, subset=['Statut'])
    except AttributeError: return df.style.applymap(styliser_statut, subset=['Statut'])

def lisser_boite(ancienne, nouvelle, alpha=0.3):
    if ancienne is None: return nouvelle
    return {
        "x1": int(ancienne["x1"] * (1 - alpha) + nouvelle["x1"] * alpha),
        "y1": int(ancienne["y1"] * (1 - alpha) + nouvelle["y1"] * alpha),
        "x2": int(ancienne["x2"] * (1 - alpha) + nouvelle["x2"] * alpha),
        "y2": int(ancienne["y2"] * (1 - alpha) + nouvelle["y2"] * alpha),
    }

def appliquer_visuels_sur_image(frame_cv2, donnees_api, memoire_boites=None):
    if not donnees_api or not donnees_api.get("success"): return frame_cv2, memoire_boites
    largeur_frame = frame_cv2.shape[1]
    
    for vehicule in donnees_api["data"]:
        plaque, statut, confiance = vehicule["plaque"], vehicule["statut"], vehicule["confiance"]
        nouvelle_bbox = vehicule["bounding_box"]
        
        enregistrer_historique(plaque, statut, confiance)
        
        boite_finale = nouvelle_bbox
        if memoire_boites is not None:
            boite_finale = lisser_boite(memoire_boites.get(plaque), nouvelle_bbox)
            memoire_boites[plaque] = boite_finale
            
        x1, y1, x2, y2 = boite_finale["x1"], boite_finale["y1"], boite_finale["x2"], boite_finale["y2"]
        couleur_bgr = (0, 255, 0) if statut == "AUTORISE" else (0, 0, 255)
        
        # Boite et étiquette de la voiture
        cv2.rectangle(frame_cv2, (x1, y1), (x2, y2), couleur_bgr, 3)
        cv2.rectangle(frame_cv2, (x1, max(0, y1 - 25)), (x1 + 180, max(0, y1)), (0, 0, 0), -1)
        cv2.putText(frame_cv2, f"{plaque}", (x1 + 5, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur_bgr, 2)
        
        # Panneau de contrôle HUD global en haut à droite
        cv2.rectangle(frame_cv2, (largeur_frame - 350, 10), (largeur_frame - 10, 80), (0, 0, 0), -1)
        cv2.putText(frame_cv2, f"PLAQUE: {plaque}", (largeur_frame - 340, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_cv2, f"STATUT: {statut}", (largeur_frame - 340, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, couleur_bgr, 2)

    return frame_cv2, memoire_boites

# ==========================================
# ONGLETS D'ANALYSE
# ==========================================
with onglet_import_img:
    col_img, col_tab_img = st.columns([7, 3])
    with col_img:
        fichier_upload = st.file_uploader("Selectionnez une image (.jpg, .png)", type=['jpg', 'jpeg', 'png'])
        if fichier_upload is not None and st.button("Analyser l'image", type="primary"):
            image_bytes = fichier_upload.getvalue()
            donnees = interroger_api(image_bytes, fichier_upload.name)
            
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame_annote, _ = appliquer_visuels_sur_image(frame, donnees)
            
            frame_rgb = cv2.cvtColor(frame_annote, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="Resultat de l'analyse", use_container_width=True)
            
    with col_tab_img:
        st.markdown("### 📋 ANALYSES EN COURS")
        st.dataframe(obtenir_tableau_style(), use_container_width=True, hide_index=True)

with onglet_import_vid:
    col_vid, col_tab_vid = st.columns([7, 3])
    with col_vid:
        video_upload = st.file_uploader("Selectionnez une video (.mp4, .mov)", type=['mp4', 'mov', 'avi'])
        btn_demarrer_vid = st.button("Demarrer l'analyse video", type="primary")
        cadre_video_import = st.empty()
        
    with col_tab_vid:
        st.markdown("### 📋 ANALYSES EN COURS")
        cadre_tableau_vid = st.empty()
        cadre_tableau_vid.dataframe(obtenir_tableau_style(), use_container_width=True, hide_index=True)
    
    if video_upload is not None and btn_demarrer_vid:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_upload.read())
        cap = cv2.VideoCapture(tfile.name)
        compteur_frames = 0
        memoire_boites_video = {}
        dernieres_donnees = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            compteur_frames += 1
            if compteur_frames % 10 == 0:
                _, buffer = cv2.imencode('.jpg', frame)
                dernieres_donnees = interroger_api(buffer.tobytes(), f"frame_{compteur_frames}.jpg")
            
            frame_annote, memoire_boites_video = appliquer_visuels_sur_image(frame, dernieres_donnees, memoire_boites_video)
            frame_rgb = cv2.cvtColor(frame_annote, cv2.COLOR_BGR2RGB)
            cadre_video_import.image(frame_rgb, channels="RGB", use_container_width=True)
            cadre_tableau_vid.dataframe(obtenir_tableau_style(), use_container_width=True, hide_index=True)
        cap.release()

with onglet_camera:
    col_cam, col_tab_cam = st.columns([7, 3])
    with col_cam:
        if st.button("Allumer / Eteindre l'Appareil Photo"):
            st.session_state.camera_active = not st.session_state.camera_active

        if st.session_state.camera_active:
            capture_camera = st.camera_input("Prendre une photo", label_visibility="collapsed")
            if capture_camera is not None and st.button("Analyser la photo", type="primary"):
                image_bytes = capture_camera.getvalue()
                donnees = interroger_api(image_bytes, "snapshot.jpg")
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                frame_annote, _ = appliquer_visuels_sur_image(frame, donnees)
                frame_rgb = cv2.cvtColor(frame_annote, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption="Detection Camera", use_container_width=True)
    with col_tab_cam:
        st.markdown("### 📋 ANALYSES EN COURS")
        st.dataframe(obtenir_tableau_style(), use_container_width=True, hide_index=True)

with onglet_stream:
    col_flux, col_tab_flux = st.columns([7, 3])
    with col_flux:
        choix_source = st.radio(
            "Selectionnez la source de la camera :",
            ("Lien RTMP / HTTP direct (ex: rtmp://stream.it-innov...)", "Application Mobile IP Webcam", "Webcam Ordinateur Locale (0)", "Camera Phone Link Locale (1)")
        )
        source = None
        
        # Logique de parsing intelligente pour eviter les plantages d'OpenCV
        if choix_source == "Lien RTMP / HTTP direct (ex: rtmp://stream.it-innov...)":
            source = st.text_input("Entrez le lien direct brut:")
            # Pour ce choix, aucune modification, on envoie exactement ce que l'utilisateur a tapé à OpenCV.
            
        elif choix_source == "Application Mobile IP Webcam":
            url_input = st.text_input("Entrez UNIQUEMENT l'adresse IP et le Port affichés sur votre téléphone (ex: 192.168.0.36:8080)")
            if url_input:
                source = url_input.strip().replace("http://", "").replace("/video", "")
                source = f"http://{source}/video" # Formatage strict obligatoire pour OpenCV
                
        elif "Webcam Ordinateur Locale (0)" in choix_source: source = 0
        elif "Camera Phone Link Locale (1)" in choix_source: source = 1

        demarrer_stream = st.checkbox("Demarrer le flux de controle en temps reel")
        cadre_video_stream = st.empty()
        
    with col_tab_flux:
        st.markdown("### 📋 ANALYSES EN COURS")
        cadre_tableau_stream = st.empty()
        cadre_tableau_stream.dataframe(obtenir_tableau_style(), use_container_width=True, hide_index=True)

    if demarrer_stream and source is not None and source != "":
        cap = cv2.VideoCapture(source)
        if not cap.isOpened(): st.error(f"Echec de connexion au flux. Verifiez le lien/reseau.")
        else:
            compteur_frames = 0
            memoire_boites_stream = {}
            dernieres_donnees = None
            
            while demarrer_stream:
                ret, frame = cap.read()
                if not ret: break
                
                compteur_frames += 1
                if compteur_frames % 10 == 0:
                    _, buffer = cv2.imencode('.jpg', frame)
                    dernieres_donnees = interroger_api(buffer.tobytes(), "stream_frame.jpg")
                    if not dernieres_donnees or not dernieres_donnees.get("success"):
                        dernieres_donnees = None
                        memoire_boites_stream.clear()
                
                frame_annote, memoire_boites_stream = appliquer_visuels_sur_image(frame, dernieres_donnees, memoire_boites_stream)
                frame_rgb = cv2.cvtColor(frame_annote, cv2.COLOR_BGR2RGB)
                cadre_video_stream.image(frame_rgb, channels="RGB", use_container_width=True)
                cadre_tableau_stream.dataframe(obtenir_tableau_style(), use_container_width=True, hide_index=True)
            cap.release()

# ==========================================
# ONGLET 5 : ADMINISTRATION DB
# ==========================================
with onglet_admin:
    st.markdown("## ⚙️ Gestion des Accès Parking")
    col_form, col_db = st.columns(2)
    
    with col_form:
        st.subheader("Action")
        nouvelle_plaque = st.text_input("Ajouter une plaque (ex: ABCD123)")
        if st.button("➕ Autoriser ce véhicule", type="primary"):
            if nouvelle_plaque:
                res = requests.post(f"{API_DB_URL}/{nouvelle_plaque}")
                if res.status_code == 200: st.success("Plaque ajoutée avec succès.")
                else: st.error("Erreur lors de l'ajout.")
        
        st.markdown("---")
        
        plaque_a_supprimer = st.text_input("Supprimer une plaque")
        if st.button("🗑️ Révoquer l'accès"):
            if plaque_a_supprimer:
                res = requests.delete(f"{API_DB_URL}/{plaque_a_supprimer}")
                if res.status_code == 200: st.warning("Accès révoqué.")
                else: st.error("Erreur lors de la suppression.")
                
    with col_db:
        st.subheader("Base de données actuelle (SQLite)")
        try:
            reponse_db = requests.get(API_DB_URL)
            if reponse_db.status_code == 200:
                data_db = reponse_db.json().get("data", [])
                if data_db:
                    df_db = pd.DataFrame(data_db)
                    st.dataframe(df_db, use_container_width=True, hide_index=True)
                else:
                    st.info("Aucune plaque enregistrée dans la base de données.")
        except:
            st.error("Impossible de se connecter à la base de données via l'API.")