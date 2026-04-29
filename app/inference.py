"""Module d'inférence — chargement du modèle et prédiction sur une image."""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
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
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    return model


def predict(model: nn.Module, image: Image.Image, device: torch.device, tta: bool = True) -> dict:
    """Prédit la classe d'une radiographie pulmonaire.

    Args:
        model:  Modèle EfficientNet-B0 chargé via :func:`load_model`.
        image:  Image PIL en mode RGB.
        device: Dispositif sur lequel exécuter l'inférence.
        tta:    Si True, moyenne les prédictions sur l'image originale et son flip horizontal
                (Test Time Augmentation). Améliore légèrement la robustesse sans réentraîner.
    """
    images = [image]
    if tta:
        images.append(TF.hflip(image))

    tensors = torch.stack([_val_transforms(img) for img in images]).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensors), dim=1).mean(dim=0)

    pred_idx = probs.argmax().item()
    classe   = CLASSES[pred_idx]

    return {
        "classe":       classe,
        "label":        LABELS_FR[classe],
        "confiance":    round(probs[pred_idx].item(), 4),
        "probabilites": {CLASSES[i]: round(probs[i].item(), 4) for i in range(len(CLASSES))},
    }
