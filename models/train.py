"""Entraînement du classifieur EfficientNet-B0 sur les radiographies pulmonaires.

Améliorations par rapport à la version initiale :
- Loss pondérée par classe pour compenser le déséquilibre du dataset
- Label smoothing pour améliorer la généralisation
- ReduceLROnPlateau : réduit le LR quand la val_loss stagne
- Early stopping : interrompt l'entraînement si pas d'amélioration
- Gradient clipping : stabilise l'entraînement
- Suivi de l'accuracy train/val en plus de la loss

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


class EarlyStopping:
    """Arrête l'entraînement si la val_loss ne s'améliore pas pendant ``patience`` époques."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = float("inf")

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def compute_class_weights(dataset: ChestXRayDataset) -> torch.Tensor:
    """Calcule les poids inversement proportionnels à la fréquence de chaque classe.

    Formule : weight_i = total / (n_classes × count_i)
    Permet de compenser le déséquilibre NORMAL / BACTERIA / VIRUS.
    """
    counts  = dataset.get_class_counts()
    total   = sum(counts.values())
    n       = len(ChestXRayDataset.CLASSES)
    weights = torch.tensor(
        [total / (n * counts[c]) for c in ChestXRayDataset.CLASSES],
        dtype=torch.float,
    )
    return weights


def build_model(unfreeze_last_n: int = 0) -> nn.Module:
    """Construit un EfficientNet-B0 pré-entraîné adapté à la classification 3 classes.

    Args:
        unfreeze_last_n: Nombre de blocs finaux du feature extractor à dégeler.
                         ``0`` → seul le classifieur est entraînable (head training).
                         ``3`` → les 3 derniers blocs + classifieur sont entraînables (fine-tuning).
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
    """Crée les DataLoaders pour les trois partitions du jeu de données."""
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
    scheduler,
    num_epochs: int = 30,
    patience: int = 7,
    save_path: str = "models/best_model.pth",
    run_name: str | None = None,
) -> None:
    """Entraîne le modèle avec early stopping et scheduler de LR.

    Args:
        model:       Modèle à entraîner.
        dataloaders: Dictionnaire ``{"train", "val", "test"}``.
        criterion:   Fonction de perte.
        optimizer:   Optimiseur PyTorch.
        scheduler:   Scheduler de LR (ex. ReduceLROnPlateau).
        num_epochs:  Nombre maximum d'époques.
        patience:    Patience de l'early stopping.
        save_path:   Chemin de sauvegarde du meilleur checkpoint.
        run_name:    Nom du run MLflow.
    """
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    early_stopping    = EarlyStopping(patience=patience)
    meilleure_val_loss = float("inf")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "num_epochs":    num_epochs,
            "batch_size":    dataloaders["train"].batch_size,
            "patience":      patience,
            "optimizer":     type(optimizer).__name__,
            "scheduler":     type(scheduler).__name__,
        })

        for epoch in range(num_epochs):

            # — Phase entraînement —
            model.train()
            train_loss    = 0.0
            train_correct = 0
            train_total   = 0

            for images, labels in dataloaders["train"]:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss   = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss    += loss.item() * images.size(0)
                train_correct += (logits.argmax(dim=1) == labels).sum().item()
                train_total   += images.size(0)

            train_loss /= train_total
            train_acc   = train_correct / train_total

            # — Phase validation —
            model.eval()
            val_loss    = 0.0
            val_correct = 0
            val_total   = 0

            with torch.no_grad():
                for images, labels in dataloaders["val"]:
                    images, labels = images.to(device), labels.to(device)
                    logits  = model(images)
                    val_loss += criterion(logits, labels).item() * images.size(0)
                    val_correct += (logits.argmax(dim=1) == labels).sum().item()
                    val_total   += images.size(0)

            val_loss /= val_total
            val_acc   = val_correct / val_total

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "train_acc":  train_acc,
                "val_acc":    val_acc,
                "lr":         current_lr,
            }, step=epoch)

            print(
                f"Époque {epoch + 1:>3}/{num_epochs} | "
                f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} acc={val_acc:.3f} | "
                f"lr={current_lr:.2e}"
            )

            if val_loss < meilleure_val_loss:
                meilleure_val_loss = val_loss
                torch.save({
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":             meilleure_val_loss,
                    "val_acc":              val_acc,
                }, save_path)
                print(f"  → Checkpoint sauvegardé (val_loss={meilleure_val_loss:.4f})")
                mlflow.log_artifact(save_path)

            if early_stopping.step(val_loss):
                print(f"\nEarly stopping déclenché à l'époque {epoch + 1}.")
                break

    print(f"\nEntraînement terminé — meilleure val_loss : {meilleure_val_loss:.4f}")


if __name__ == "__main__":
    data_path = os.getenv("DATA_PATH")

    FINETUNE   = True
    CHECKPOINT = "models/best_model.pth"
    SAVE_PATH  = "models/best_model_ft.pth"
    NUM_EPOCHS = 30
    LR         = 1e-4
    PATIENCE   = 7

    model = build_model(unfreeze_last_n=3 if FINETUNE else 0).to(device)

    if FINETUNE and os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(
            f"Poids chargés depuis {CHECKPOINT} "
            f"(époque {ckpt['epoch'] + 1}, val_loss={ckpt['val_loss']:.4f})"
        )
    elif FINETUNE:
        print(f"Checkpoint introuvable ({CHECKPOINT}) — fine-tuning depuis zéro.")

    dataloaders    = get_dataloaders(data_path)
    class_weights  = compute_class_weights(dataloaders["train"].dataset).to(device)
    criterion      = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer      = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    run_name = "efficientnet_b0_finetune_v2" if FINETUNE else "efficientnet_b0_head_v2"

    train_model(
        model, dataloaders, criterion, optimizer, scheduler,
        num_epochs=NUM_EPOCHS, patience=PATIENCE,
        save_path=SAVE_PATH, run_name=run_name,
    )
