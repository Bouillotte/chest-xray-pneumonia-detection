"""Visualisation Grad-CAM pour l'interprétabilité des prédictions du modèle."""

import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from data.transforms import get_val_transforms, IMAGENET_MEAN, IMAGENET_STD  # noqa: F401

_CLASSES = ["NORMAL", "BACTERIA", "VIRUS"]
_preprocess = get_val_transforms()


def generate_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    image_np: np.ndarray,
) -> np.ndarray:
    """Génère un overlay Grad-CAM sur une image prétraitée.

    Args:
        model:        Modèle EfficientNet-B0 en mode évaluation.
        image_tensor: Tenseur de forme ``(1, 3, 224, 224)`` normalisé ImageNet.
        image_np:     Image redimensionnée de forme ``(224, 224, 3)``,
                      valeurs en virgule flottante dans ``[0, 1]``.

    Returns:
        Tableau uint8 ``(224, 224, 3)`` avec la heatmap Grad-CAM superposée.
    """
    for param in model.parameters():
        param.requires_grad_(True)

    cam = GradCAM(model=model, target_layers=[model.features[-1]])
    masque = cam(input_tensor=image_tensor)[0]
    return show_cam_on_image(image_np, masque, use_rgb=True)


def generate_gradcam_pil(model: nn.Module, image_pil: Image.Image) -> Image.Image:
    """Génère un overlay Grad-CAM à partir d'une image PIL.

    Wrapper adapté à Streamlit : entrée et sortie au format ``PIL.Image``.

    Args:
        model:     Modèle EfficientNet-B0 chargé via ``load_model``.
        image_pil: Radiographie originale en mode RGB.

    Returns:
        Image PIL ``(224, 224)`` avec la heatmap Grad-CAM superposée.
    """
    image_resized = image_pil.resize((224, 224)).convert("RGB")
    image_np = np.array(image_resized, dtype=np.float32) / 255.0
    image_tensor = _preprocess(image_pil).unsqueeze(0)

    overlay = generate_gradcam(model, image_tensor, image_np)
    return Image.fromarray(overlay)


def save_gradcam(
    overlay: np.ndarray,
    image_original: np.ndarray,
    label_pred: int,
    label_true: int,
    save_path: str,
) -> None:
    """Sauvegarde une figure côte à côte : image originale et overlay Grad-CAM.

    Args:
        overlay:        Tableau uint8 ``(H, W, 3)`` produit par :func:`generate_gradcam`.
        image_original: Image originale ``(H, W, 3)``.
        label_pred:     Indice de classe prédit (0 = NORMAL, 1 = BACTERIA, 2 = VIRUS).
        label_true:     Indice de classe réel.
        save_path:      Chemin de sauvegarde du fichier image.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(image_original)
    axes[0].set_title(f"Réel : {_CLASSES[label_true]}")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(f"Prédit : {_CLASSES[label_pred]}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
