"""Module d'inférence — chargement du modèle et prédiction sur une image."""

import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image

from data.transforms import get_val_transforms

CLASSES = ["NORMAL", "BACTERIA", "VIRUS"]

LABELS_FR: dict[str, str] = {
    "NORMAL":   "✅ Poumon sain",
    "BACTERIA": "🦠 Pneumonie bactérienne",
    "VIRUS":    "⚠️ Pneumonie virale",
}

_val_transforms = get_val_transforms()


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Charge un EfficientNet-B0 fine-tuné depuis un checkpoint PyTorch.

    Args:
        checkpoint_path: Chemin vers le fichier ``.pth`` produit lors de l'entraînement.
        device:          Dispositif cible (CPU ou CUDA).

    Returns:
        Modèle en mode évaluation, prêt à l'inférence.
    """
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    return model


def predict(model: nn.Module, image: Image.Image, device: torch.device) -> dict:
    """Prédit la classe d'une radiographie pulmonaire.

    Args:
        model:  Modèle EfficientNet-B0 chargé via :func:`load_model`.
        image:  Image PIL en mode RGB.
        device: Dispositif sur lequel exécuter l'inférence.

    Returns:
        Dictionnaire contenant :

        - ``"classe"``       — identifiant de la classe prédite (ex. ``"BACTERIA"``).
        - ``"label"``        — libellé avec emoji (ex. ``"🦠 Pneumonie bactérienne"``).
        - ``"confiance"``    — score softmax de la classe prédite (entre 0 et 1).
        - ``"probabilites"`` — scores softmax pour chacune des 3 classes.
    """
    tensor = _val_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()

    pred_idx = probs.argmax().item()
    classe = CLASSES[pred_idx]

    return {
        "classe":       classe,
        "label":        LABELS_FR[classe],
        "confiance":    round(probs[pred_idx].item(), 4),
        "probabilites": {CLASSES[i]: round(probs[i].item(), 4) for i in range(len(CLASSES))},
    }
