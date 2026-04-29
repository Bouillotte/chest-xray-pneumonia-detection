"""Point d'entrée Streamlit pour l'exécution en développement local.

Résout les chemins via PROJECT_PATH dans le fichier .env.
Utilisation : streamlit run app/streamlit_app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st
import torch
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
PROJECT_PATH = Path(os.getenv("PROJECT_PATH", str(Path(__file__).parent.parent)))
sys.path.append(str(PROJECT_PATH))

from app.inference import load_model, predict, LABELS_FR
from utils.gradcam import generate_gradcam_pil

st.set_page_config(
    page_title="Détection de pneumonie · EfficientNet-B0",
    page_icon="🫁",
    layout="wide",
)

MODEL_PATH  = PROJECT_PATH / "models" / "best_model.pth"
SAMPLES_DIR = PROJECT_PATH / "samples"
DEVICE      = torch.device("cpu")

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


@st.cache_data
def analyser_fichier(chemin: str):
    image    = Image.open(chemin).convert("RGB")
    resultat = predict(_model, image, DEVICE)
    gradcam  = generate_gradcam_pil(_model, image)
    return image, resultat, gradcam


def afficher_resultat(image: Image.Image, resultat: dict, gradcam: Image.Image) -> None:
    _AFFICHAGE_CLASSE[resultat["classe"]](
        f"**{resultat['label']}** · {resultat['confiance'] * 100:.1f} % de confiance"
    )
    col_img, col_cam = st.columns(2)

    with col_img:
        st.caption("Radiographie originale")
        st.image(image, use_container_width=True)

    with col_cam:
        st.caption("Carte d'activation Grad-CAM")
        st.image(gradcam, use_container_width=True)
        st.caption("Les zones rouges correspondent aux régions déterminantes pour la prédiction.")
        st.markdown("**Probabilités par classe**")
        for classe, prob in resultat["probabilites"].items():
            st.progress(float(prob), text=f"{LABELS_FR[classe]} — {prob * 100:.1f} %")


_model = charger_modele()

st.title("🫁 Détection de Pneumonie par Radiographie Thoracique")
st.markdown("Classification automatique · **EfficientNet-B0** · 3 classes · PyTorch · Grad-CAM")
st.divider()

if _model is None:
    st.error(f"Modèle introuvable : `{MODEL_PATH}`")
    st.stop()

tab_about, tab_examples, tab_demo = st.tabs(["À propos", "Exemples", "Démo"])

with tab_about:
    col_desc, col_tech = st.columns([3, 2])

    with col_desc:
        st.subheader("Présentation")
        st.markdown("""
Ce projet est un classifieur de radiographies pulmonaires capable de distinguer trois situations cliniques :

| Classe | Description |
|---|---|
| **NORMAL** | Poumon sain, pas de pathologie détectée |
| **BACTERIA** | Pneumonie bactérienne — consolidations lobaires |
| **VIRUS** | Pneumonie virale — infiltrats bilatéraux diffus |

#### Démarche

Le modèle repose sur **EfficientNet-B0** pré-entraîné sur ImageNet (1,2 M d'images).
L'entraînement suit une stratégie en deux phases :

1. **Head training** — le backbone est gelé, seule la tête de classification est entraînée
2. **Fine-tuning** — les 3 derniers blocs convolutifs sont dégelés pour s'adapter aux
   caractéristiques visuelles propres aux radiographies pulmonaires

La fonction de perte est **pondérée par classe** pour compenser le déséquilibre du dataset
(environ 3× plus de cas de pneumonie que de cas normaux).

#### Interprétabilité

Chaque prédiction est accompagnée d'une **carte Grad-CAM** qui identifie les zones de la
radiographie ayant le plus influencé la décision — un critère essentiel pour évaluer la
pertinence clinique du modèle.
        """)

    with col_tech:
        st.subheader("Stack technique")
        st.markdown("""
**Modèle**
- EfficientNet-B0 · torchvision
- Fine-tuning PyTorch
- Grad-CAM · pytorch-grad-cam

**Données**
- [Chest X-Ray Images](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) · Kaggle
- 5 856 radiographies
- 3 classes : NORMAL / BACTERIA / VIRUS

**Déploiement**
- Interface : Streamlit
- API REST : FastAPI + Uvicorn
- Conteneurisation : Docker
- Hébergement : HuggingFace Spaces
        """)

    st.divider()
    st.caption(
        "⚠️ Prototype de recherche — ne remplace pas un diagnostic médical "
        "établi par un professionnel de santé qualifié."
    )

with tab_examples:
    st.subheader("Exemples de prédictions")
    st.markdown("Résultats du modèle sur trois radiographies représentatives, une par classe.")

    SAMPLES = {
        "NORMAL":   SAMPLES_DIR / "NORMAL.jpeg",
        "BACTERIA": SAMPLES_DIR / "BACTERIA.jpeg",
        "VIRUS":    SAMPLES_DIR / "VIRUS.jpeg",
    }

    manquants = [k for k, p in SAMPLES.items() if not p.exists()]
    if manquants:
        st.info(
            f"Images d'exemple introuvables ({', '.join(manquants)}). "
            "Ajoute `NORMAL.jpeg`, `BACTERIA.jpeg` et `VIRUS.jpeg` dans le dossier `samples/`."
        )
    else:
        for classe, path in SAMPLES.items():
            st.markdown(f"#### {LABELS_FR[classe]}")
            with st.spinner("Analyse…"):
                image, resultat, gradcam = analyser_fichier(str(path))
            afficher_resultat(image, resultat, gradcam)
            st.divider()

with tab_demo:
    st.subheader("Tester le modèle")

    samples_dispo = {
        k: p
        for k, p in {
            "NORMAL":   SAMPLES_DIR / "NORMAL.jpeg",
            "BACTERIA": SAMPLES_DIR / "BACTERIA.jpeg",
            "VIRUS":    SAMPLES_DIR / "VIRUS.jpeg",
        }.items()
        if p.exists()
    }

    col_upload, col_sample = st.columns(2)
    image_a_analyser = None

    with col_upload:
        st.markdown("**Déposer une radiographie**")
        fichier = st.file_uploader(
            "Format accepté : JPG ou PNG",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if fichier:
            image_a_analyser = Image.open(fichier).convert("RGB")

    with col_sample:
        if samples_dispo:
            st.markdown("**Ou choisir un exemple**")
            choix = st.radio(
                "",
                list(samples_dispo.keys()),
                format_func=lambda k: LABELS_FR[k],
                label_visibility="collapsed",
            )
            if st.button("Analyser cet exemple", use_container_width=True):
                image_a_analyser = Image.open(samples_dispo[choix]).convert("RGB")

    if image_a_analyser is not None:
        st.divider()
        with st.spinner("Analyse en cours…"):
            resultat = predict(_model, image_a_analyser, DEVICE)
            gradcam  = generate_gradcam_pil(_model, image_a_analyser)
        afficher_resultat(image_a_analyser, resultat, gradcam)

    st.divider()
    st.caption(
        "⚠️ Prototype de recherche — ne remplace pas un diagnostic médical "
        "établi par un professionnel de santé qualifié."
    )
