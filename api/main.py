"""API REST FastAPI : Prédiction de pneumonie sur radiographie pulmonaire.

Routes :
    GET  /         — Vérification de disponibilité.
    GET  /health   — État du service et du modèle chargé.
    POST /predict  — Prédiction à partir d'une image JPEG ou PNG.
"""

import io
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

load_dotenv()
project_path = os.getenv("PROJECT_PATH", "/app")
sys.path.append(project_path)

from app.inference import load_model, predict

MODEL_PATH      = Path(project_path) / "models" / "best_model.pth"
DEVICE          = torch.device("cpu")
TYPES_ACCEPTES  = {"image/jpeg", "image/jpg", "image/png"}

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage et libère les ressources à l'arrêt."""
    global _model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modèle introuvable : {MODEL_PATH}")
    _model = load_model(str(MODEL_PATH), DEVICE)
    yield


app = FastAPI(
    title="API de Détection de Pneumonie",
    description="Classification de radiographies pulmonaires — EfficientNet-B0 · 3 classes",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReponsePrediction(BaseModel):
    classe: str
    label: str
    confiance: float
    probabilites: dict[str, float]


@app.get("/")
def racine():
    return {"status": "ok", "message": "API de détection de pneumonie — POST /predict"}


@app.get("/health")
def sante():
    return {"status": "ok", "modele_charge": _model is not None}


@app.post("/predict", response_model=ReponsePrediction)
async def predire(fichier: UploadFile = File(...)):
    """Prédit la classe d'une radiographie pulmonaire.

    Args:
        fichier: Image JPEG ou PNG de la radiographie (multipart/form-data).

    Returns:
        JSON contenant la classe prédite, le libellé, la confiance et les
        probabilités pour chacune des 3 classes.

    Raises:
        HTTPException 400: Format d'image non supporté ou fichier illisible.
    """
    if fichier.content_type not in TYPES_ACCEPTES:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté : {fichier.content_type}. Formats acceptés : JPEG, PNG.",
        )

    contenu = await fichier.read()

    try:
        image = Image.open(io.BytesIO(contenu)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Impossible de décoder l'image.")

    return ReponsePrediction(**predict(_model, image, DEVICE))
