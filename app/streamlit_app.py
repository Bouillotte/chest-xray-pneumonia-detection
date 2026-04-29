"""Point d'entrée Streamlit pour l'exécution en développement local.

Contrairement à ``app.py`` (entrée HuggingFace Spaces / Docker), ce script
résout les chemins via la variable d'environnement ``PROJECT_PATH`` définie
dans un fichier ``.env`` à la racine du projet.

Utilisation :
    streamlit run app/streamlit_app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st
import torch
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
sys.path.append(os.getenv("PROJECT_PATH", str(Path(__file__).parent.parent)))

from app.inference import load_model, predict, LABELS_FR
from utils.gradcam import generate_gradcam_pil

st.set_page_config(
    page_title="Détection de pneumonie",
    page_icon="🫁",
    layout="wide",
)

MODEL_PATH = Path(os.getenv("PROJECT_PATH", ".")) / "models" / "best_model.pth"
DEVICE     = torch.device("cpu")

_AFFICHAGE_CLASSE = {
    "NORMAL":   st.success,
    "BACTERIA": st.error,
    "VIRUS":    st.warning,
}


@st.cache_resource
def charger_modele():
    if not MODEL_PATH.exists():
        return None
    return load_model(str(MODEL_PATH), DEVICE)


model = charger_modele()

st.title("🫁 Détection de Pneumonie")
st.markdown("Analyse de radiographies pulmonaires · EfficientNet-B0 · 3 classes · PyTorch")
st.divider()

if model is None:
    st.warning("Modèle non disponible.")
    st.info(f"Le fichier `{MODEL_PATH}` est introuvable.")
    st.stop()

fichier_upload = st.file_uploader(
    "Dépose une radiographie pulmonaire (.jpg / .png)",
    type=["jpg", "jpeg", "png"],
)

if fichier_upload is not None:
    image = Image.open(fichier_upload).convert("RGB")

    with st.spinner("Analyse en cours…"):
        resultat    = predict(model, image, DEVICE)
        img_gradcam = generate_gradcam_pil(model, image)

    # — Résultat principal —
    _AFFICHAGE_CLASSE[resultat["classe"]](f"## {resultat['label']}")
    st.metric("Confiance", f"{resultat['confiance'] * 100:.1f}%")
    st.divider()

    # — Colonnes : radiographie | Grad-CAM | probabilités —
    col_radio, col_cam, col_proba = st.columns(3)

    with col_radio:
        st.subheader("📷 Radiographie originale")
        st.image(image, use_container_width=True)

    with col_cam:
        st.subheader("🔥 Carte d'activation Grad-CAM")
        st.image(img_gradcam, use_container_width=True)
        st.caption("Les zones en rouge correspondent aux régions déterminantes pour la prédiction.")

    with col_proba:
        st.subheader("📊 Probabilités par classe")
        for classe, prob in resultat["probabilites"].items():
            st.progress(float(prob), text=f"{LABELS_FR[classe]} — {prob * 100:.1f}%")
            st.markdown("")

    st.divider()
    st.caption(
        "⚠️ Cet outil est un prototype de recherche. "
        "Il ne remplace pas un diagnostic médical établi par un professionnel de santé qualifié."
    )

st.divider()
st.caption("Projet Deep Learning · EfficientNet-B0 · 3 classes · PyTorch")
