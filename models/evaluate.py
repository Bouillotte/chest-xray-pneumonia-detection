"""Évaluation du modèle entraîné sur le jeu de test.

Génère un rapport de classification complet, une matrice de confusion et des
courbes ROC par classe (approche One-vs-Rest). Les figures sont sauvegardées
dans le répertoire ``results/``.

Variables d'environnement requises (fichier ``.env``) :
    PROJECT_PATH : répertoire racine du projet.
    DATA_PATH    : répertoire racine du jeu de données.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as models
from dotenv import load_dotenv
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

load_dotenv()
sys.path.append(os.getenv("PROJECT_PATH", "."))

from data.dataset import ChestXRayDataset
from data.transforms import get_val_transforms

CLASSES = ["NORMAL", "BACTERIA", "VIRUS"]
device  = torch.device("cpu")


def load_model(checkpoint_path: str) -> nn.Module:
    """Charge un EfficientNet-B0 fine-tuné depuis un fichier checkpoint.

    Args:
        checkpoint_path: Chemin vers le fichier ``.pth`` produit par :mod:`models.train`.

    Returns:
        Modèle en mode évaluation.
    """
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Modèle chargé — époque {ckpt['epoch'] + 1} | val_loss {ckpt['val_loss']:.4f}")
    return model


def get_predictions(
    model: nn.Module,
    dataloader: DataLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collecte les prédictions du modèle sur l'ensemble d'un DataLoader.

    Args:
        model:      Modèle en mode évaluation.
        dataloader: DataLoader du jeu de test.

    Returns:
        Triplet ``(labels, preds, probs)`` sous forme de tableaux NumPy :

        - ``labels`` — labels réels, forme ``(N,)``.
        - ``preds``  — classes prédites (argmax softmax), forme ``(N,)``.
        - ``probs``  — scores softmax, forme ``(N, nb_classes)``.
    """
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in dataloader:
            probs = torch.softmax(model(images.to(device)), dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    save_path: str = "results/confusion_matrix.png",
) -> None:
    """Sauvegarde la matrice de confusion sous forme de heatmap annotée.

    Args:
        labels:    Labels réels.
        preds:     Prédictions du modèle.
        save_path: Chemin de sauvegarde de la figure.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title("Matrice de confusion — Jeu de test")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Matrice de confusion : {save_path}")


def plot_roc_curves(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: str = "results/roc_curves.png",
) -> None:
    """Sauvegarde les courbes ROC (une courbe par classe, approche One-vs-Rest).

    Args:
        labels:    Labels réels.
        probs:     Scores softmax de forme ``(N, nb_classes)``.
        save_path: Chemin de sauvegarde de la figure.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    couleurs = ["steelblue", "tomato", "seagreen"]

    plt.figure(figsize=(8, 6))
    for i, (cls, couleur) in enumerate(zip(CLASSES, couleurs)):
        labels_binaires = (labels == i).astype(int)
        fpr, tpr, _ = roc_curve(labels_binaires, probs[:, i])
        auc = roc_auc_score(labels_binaires, probs[:, i])
        plt.plot(fpr, tpr, color=couleur, label=f"{cls} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Aléatoire")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Courbes ROC — Jeu de test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Courbes ROC : {save_path}")


if __name__ == "__main__":
    data_path    = os.getenv("DATA_PATH")
    project_path = os.getenv("PROJECT_PATH", ".")
    checkpoint   = os.path.join(project_path, "models", "best_model.pth")

    model = load_model(checkpoint)

    test_dataset = ChestXRayDataset(data_path, "test", get_val_transforms())
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"Jeu de test : {len(test_dataset)} images")

    labels, preds, probs = get_predictions(model, test_loader)

    print("\n── Rapport de classification ──────────────────────────────────────")
    print(classification_report(labels, preds, target_names=CLASSES))

    scores_f1 = f1_score(labels, preds, average=None)
    for cls, score in zip(CLASSES, scores_f1):
        print(f"F1 {cls:<10} : {score:.4f}")

    auc_macro = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    print(f"\nAUC-ROC macro : {auc_macro:.4f}")

    plot_confusion_matrix(labels, preds)
    plot_roc_curves(labels, probs)

    print("\nÉvaluation terminée — résultats dans results/")
