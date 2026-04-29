"""Entraînement du classifieur EfficientNet-B0 sur les radiographies pulmonaires.

Deux stratégies sont supportées via le flag ``FINETUNE`` dans le bloc ``__main__`` :

- **Head training** (``FINETUNE=False``) — seul le classifieur est entraînable,
  le feature extractor reste gelé.
- **Fine-tuning** (``FINETUNE=True``) — les ``N`` derniers blocs du feature extractor
  sont dégelés, les poids d'un run précédent sont chargés depuis ``CHECKPOINT``.

Variables d'environnement requises (fichier ``.env``) :
    PROJECT_PATH : répertoire racine du projet (ajouté à ``sys.path``).
    DATA_PATH    : répertoire racine du jeu de données.
"""

import os
import sys

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from dotenv import load_dotenv
from torch.utils.data import DataLoader

load_dotenv()
sys.path.append(os.getenv("PROJECT_PATH", "."))

from data.dataset import ChestXRayDataset
from data.transforms import get_train_transforms, get_val_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(unfreeze_last_n: int = 0) -> nn.Module:
    """Construit un EfficientNet-B0 pré-entraîné adapté à la classification 3 classes.

    Args:
        unfreeze_last_n: Nombre de blocs finaux du feature extractor à dégeler.
                         ``0`` → seul le classifieur est entraînable (head training).
                         ``3`` → les 3 derniers blocs + classifieur sont entraînables (fine-tuning).

    Returns:
        Modèle PyTorch avec les paramètres correctement gelés/dégelés.
    """
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    for param in model.features.parameters():
        param.requires_grad = False

    if unfreeze_last_n > 0:
        for param in model.features[-unfreeze_last_n:].parameters():
            param.requires_grad = True

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    return model


def get_dataloaders(data_path: str, batch_size: int = 32) -> dict[str, DataLoader]:
    """Crée les DataLoaders pour les trois partitions du jeu de données.

    Args:
        data_path:  Répertoire racine du jeu de données.
        batch_size: Taille des mini-lots.

    Returns:
        Dictionnaire ``{"train": DataLoader, "val": DataLoader, "test": DataLoader}``.
    """
    partitions = {
        "train": (ChestXRayDataset(data_path, "train", get_train_transforms()), True),
        "val":   (ChestXRayDataset(data_path, "val",   get_val_transforms()),   False),
        "test":  (ChestXRayDataset(data_path, "test",  get_val_transforms()),   False),
    }
    return {
        nom: DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
        for nom, (ds, shuffle) in partitions.items()
    }


def train_model(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 30,
    save_path: str = "models/best_model.pth",
    run_name: str | None = None,
) -> None:
    """Entraîne le modèle et sauvegarde le meilleur checkpoint (validation loss minimale).

    Les métriques ``train_loss`` et ``val_loss`` sont enregistrées dans MLflow
    à chaque époque. Le checkpoint est sauvegardé dès qu'un nouveau minimum
    de validation loss est atteint.

    Args:
        model:       Modèle à entraîner.
        dataloaders: Dictionnaire retourné par :func:`get_dataloaders`.
        criterion:   Fonction de perte (typiquement ``CrossEntropyLoss``).
        optimizer:   Optimiseur PyTorch.
        num_epochs:  Nombre total d'époques.
        save_path:   Chemin de sauvegarde du meilleur checkpoint.
        run_name:    Nom de l'expérience MLflow (optionnel).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    meilleure_val_loss = float("inf")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "num_epochs": num_epochs,
            "batch_size": dataloaders["train"].batch_size,
        })

        for epoch in range(num_epochs):

            # — Phase entraînement —
            model.train()
            train_loss = 0.0
            for images, labels in dataloaders["train"]:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)
            train_loss /= len(dataloaders["train"].dataset)

            # — Phase validation —
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in dataloaders["val"]:
                    images, labels = images.to(device), labels.to(device)
                    val_loss += criterion(model(images), labels).item() * images.size(0)
            val_loss /= len(dataloaders["val"].dataset)

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            print(f"Époque {epoch + 1:>3}/{num_epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

            if val_loss < meilleure_val_loss:
                meilleure_val_loss = val_loss
                torch.save({
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":             meilleure_val_loss,
                }, save_path)
                print(f"  → Checkpoint sauvegardé (val_loss={meilleure_val_loss:.4f})")
                mlflow.log_artifact(save_path)

    print(f"\nEntraînement terminé — meilleure val_loss : {meilleure_val_loss:.4f}")


if __name__ == "__main__":
    data_path = os.getenv("DATA_PATH")

    FINETUNE   = True
    CHECKPOINT = "models/best_model.pth"
    SAVE_PATH  = "models/best_model_ft.pth"
    NUM_EPOCHS = 15
    LR         = 1e-4

    model = build_model(unfreeze_last_n=3 if FINETUNE else 0).to(device)

    if FINETUNE and os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(
            f"Poids chargés depuis {CHECKPOINT} "
            f"(époque {ckpt['epoch'] + 1}, val_loss={ckpt['val_loss']:.4f})"
        )
    elif FINETUNE:
        print(f"Checkpoint introuvable ({CHECKPOINT}) — fine-tuning depuis zéro avec unfreeze_last_n=3")

    dataloaders = get_dataloaders(data_path)
    criterion   = nn.CrossEntropyLoss()
    optimizer   = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
    )
    run_name = "efficientnet_b0_finetune_unfreeze3" if FINETUNE else "efficientnet_b0_frozen"

    train_model(
        model, dataloaders, criterion, optimizer,
        num_epochs=NUM_EPOCHS, save_path=SAVE_PATH, run_name=run_name,
    )
