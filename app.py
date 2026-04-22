import streamlit as st
import requests
from PIL import Image, ImageDraw
import io
import cv2
import numpy as np
import os
import tempfile

# --- 1. CONFIGURATION DYNAMIQUE ---
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1/analyze/image")

st.set_page_config(
    page_title="Controle Parking - La Cite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# INJECTION CSS : CONTROLE DE LA TAILLE DE L'ECRAN
# ==========================================
st.markdown("""
    <style>
    /* Cette regle force toutes les images et videos a ne jamais depasser 65% 
       de la hauteur de l'ecran (65vh). 'object-fit: contain' preserve les 
       proportions sans ecraser l'image, et la centre automatiquement.
    */
    [data-testid="stImage"] img {
        max-height: 65vh !important;
        object-fit: contain !important;
        margin: 0 auto !important;
        display: block !important;
    }
    /* Limite aussi la fenetre de la camera native pour eviter le debordement */
    [data-testid="stCameraInput"] video {
        max-height: 60vh !important;
    }
    </style>
""", unsafe_allow_html=True)

if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

# --- 2. INTERFACE UTILISATEUR ---
st.title("Systeme Intelligent de Controle de Parking")
st.markdown("---")

st.sidebar.header("Etat du Systeme")
st.sidebar.success("Interface operationnelle")

onglet_import_img, onglet_import_vid, onglet_camera, onglet_stream = st.tabs([
    "Importer Image", 
    "Importer Video",
    "Snapshot (Appareil Photo)", 
    "Flux Continu (Reseau & Phone Link)"
])

def interroger_api(image_bytes, nom_fichier):
    try:
        files = {"file": (nom_fichier, image_bytes, "image/jpeg")}
        reponse = requests.post(API_URL, files=files, timeout=3) 
        if reponse.status_code == 200:
            return reponse.json()
        return None
    except requests.exceptions.RequestException:
        return None

# ==========================================
# ONGLET 1 : IMPORTATION IMAGE
# ==========================================
with onglet_import_img:
    fichier_upload = st.file_uploader("Selectionnez une image (.jpg, .png)", type=['jpg', 'jpeg', 'png'])
    if fichier_upload is not None and st.button("Analyser l'image", type="primary"):
        image_bytes = fichier_upload.getvalue()
        donnees = interroger_api(image_bytes, fichier_upload.name)
        
        if donnees and donnees["success"]:
            image_pil = Image.open(io.BytesIO(image_bytes))
            draw = ImageDraw.Draw(image_pil)
            for vehicule in donnees["data"]:
                couleur = "green" if vehicule["statut"] == "AUTORISE" else "red"
                bbox = vehicule["bounding_box"]
                draw.rectangle([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]], outline=couleur, width=5)
            st.image(image_pil, caption="Resultat de l'analyse", use_container_width=True)
            for v in donnees["data"]:
                st.write(f"**Plaque:** {v['plaque']} | **Statut:** {v['statut']} (Conf: {v['confiance']:.2f})")

# ==========================================
# ONGLET 2 : IMPORTATION VIDEO
# ==========================================
with onglet_import_vid:
    st.info("Televersez une video pre-enregistree. Le systeme l'analysera image par image.")
    video_upload = st.file_uploader("Selectionnez une video (.mp4, .mov)", type=['mp4', 'mov', 'avi'])
    
    if video_upload is not None and st.button("Demarrer l'analyse video", type="primary"):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_upload.read())
        
        cap = cv2.VideoCapture(tfile.name)
        cadre_video = st.empty()
        zone_texte = st.empty()
        
        compteur_frames = 0
        dernieres_detections = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.success("Analyse video terminee.")
                break
            
            compteur_frames += 1
            
            if compteur_frames % 10 == 0:
                _, buffer = cv2.imencode('.jpg', frame)
                image_bytes = buffer.tobytes()
                donnees = interroger_api(image_bytes, f"frame_{compteur_frames}.jpg")
                
                if donnees and donnees["success"]:
                    dernieres_detections = donnees["data"]
                    plaque_info = dernieres_detections[0]
                    zone_texte.success(f"Detection a la trame {compteur_frames} : {plaque_info['plaque']} - {plaque_info['statut']}")
                else:
                    dernieres_detections = []
            
            for vehicule in dernieres_detections:
                bbox = vehicule["bounding_box"]
                statut = vehicule["statut"]
                plaque = vehicule["plaque"]
                couleur_cv2 = (0, 255, 0) if statut == "AUTORISE" else (0, 0, 255)
                
                cv2.rectangle(frame, (bbox["x1"], bbox["y1"]), (bbox["x2"], bbox["y2"]), couleur_cv2, 3)
                cv2.putText(frame, f"{plaque} ({statut})", (bbox["x1"], max(0, bbox["y1"] - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, couleur_cv2, 2)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cadre_video.image(frame_rgb, channels="RGB", use_container_width=True)
            
        cap.release()

# ==========================================
# ONGLET 3 : SNAPSHOT
# ==========================================
with onglet_camera:
    st.write("Controlez l'allumage de la camera pour capturer un instantane.")
    if st.button("Allumer / Eteindre l'Appareil Photo"):
        st.session_state.camera_active = not st.session_state.camera_active

    if st.session_state.camera_active:
        capture_camera = st.camera_input("Prendre une photo", label_visibility="collapsed")
        if capture_camera is not None and st.button("Analyser la photo", type="primary"):
            image_bytes = capture_camera.getvalue()
            donnees = interroger_api(image_bytes, "snapshot.jpg")
            
            if donnees and donnees["success"]:
                image_pil = Image.open(io.BytesIO(image_bytes))
                draw = ImageDraw.Draw(image_pil)
                for vehicule in donnees["data"]:
                    couleur = "green" if vehicule["statut"] == "AUTORISE" else "red"
                    bbox = vehicule["bounding_box"]
                    draw.rectangle([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]], outline=couleur, width=5)
                st.image(image_pil, caption="Detection Camera", use_container_width=True)
    else:
        st.info("L'appareil photo est eteint. Cliquez sur le bouton pour l'allumer.")

# ==========================================
# ONGLET 4 : FLUX VIDEO CONTINU
# ==========================================
with onglet_stream:
    st.markdown("""
    **Remarque d'Architecture :** Les webcams locales (Options 0, 1 et 2) ne fonctionnent que si Streamlit est lance localement. Si vous utilisez Docker, utilisez exclusivement le Lien IP.
    """)
    
    choix_source = st.radio(
        "Selectionnez la source de la camera :",
        (
            "Lien IP (Streaming Reseau - IP Webcam)", 
            "Webcam Ordinateur Locale (0)",
            "Camera Phone Link Locale Principale (1)",
            "Camera Phone Link Locale Secondaire (2)"
        )
    )
    
    source = None
    
    if choix_source == "Lien IP (Streaming Reseau - IP Webcam)":
        url_input = st.text_input("Entrez l'URL du flux video IP Webcam (ex: 192.168.0.36:8080/video)")
        if url_input:
            # Correction automatique : Ajout de http:// si l'utilisateur l'oublie
            source = url_input if url_input.startswith("http://") or url_input.startswith("https://") else f"http://{url_input}"
    elif "Webcam Ordinateur Locale (0)" in choix_source:
        source = 0
    elif "Camera Phone Link Locale Principale (1)" in choix_source:
        source = 1
    elif "Camera Phone Link Locale Secondaire (2)" in choix_source:
        source = 2

    demarrer_stream = st.checkbox("Demarrer le flux de controle en temps reel")
    
    cadre_video = st.empty()
    zone_texte = st.empty()

    if demarrer_stream and source is not None:
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            st.error(f"Connexion echouee a la source : {source}. Verifiez le reseau ou les autorisations de la camera.")
        else:
            compteur_frames = 0
            dernieres_detections = []
            
            while demarrer_stream:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Perte du signal video.")
                    break
                
                compteur_frames += 1
                
                if compteur_frames % 10 == 0:
                    _, buffer = cv2.imencode('.jpg', frame)
                    image_bytes = buffer.tobytes()
                    
                    donnees = interroger_api(image_bytes, "stream_frame.jpg")
                    
                    if donnees and donnees["success"]:
                        dernieres_detections = donnees["data"]
                        plaque_info = dernieres_detections[0]
                        zone_texte.success(f"Derniere detection : {plaque_info['plaque']} - {plaque_info['statut']}")
                    else:
                        dernieres_detections = []
                
                for vehicule in dernieres_detections:
                    bbox = vehicule["bounding_box"]
                    statut = vehicule["statut"]
                    plaque = vehicule["plaque"]
                    couleur_cv2 = (0, 255, 0) if statut == "AUTORISE" else (0, 0, 255)
                    
                    cv2.rectangle(frame, (bbox["x1"], bbox["y1"]), (bbox["x2"], bbox["y2"]), couleur_cv2, 3)
                    cv2.putText(frame, f"{plaque} ({statut})", (bbox["x1"], max(0, bbox["y1"] - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, couleur_cv2, 2)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cadre_video.image(frame_rgb, channels="RGB", use_container_width=True)
                
            cap.release()