"""Entraînement EfficientNet-B0 — Version 3

Améliorations majeures par rapport à la version précédente :

1. Split stratifié 85/15 depuis train+val
   Le val officiel Kaggle = 16 images seulement → signal de validation inutilisable.
   On fusionne train+val et on recrée un split propre avec sklearn.

2. Entraînement en deux phases automatiques
   Phase 1 : backbone gelé, tête seule, LR élevé  →  converge vite et proprement
   Phase 2 : 3 derniers blocs dégelés, LR différentiel  →  adaptation fine

3. AdamW + weight decay  (meilleure régularisation qu'Adam)

4. OneCycleLR par phase  (warmup + cosine annealing, plus stable que ReduceLROnPlateau)

5. Mixup en phase 2  (régularisation puissante, réduit l'overconfidence)

6. Dropout augmenté à 0.4  (EfficientNet-B0 par défaut : 0.2)

7. Early stopping et sauvegarde sur val_acc  (plus fiable que val_loss sur données déséquilibrées)

Variables d'environnement requises (.env) :
    PROJECT_PATH : répertoire racine du projet
    DATA_PATH    : répertoire racine du dataset
"""

import os
import sys
from collections import Counter
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from dotenv import load_dotenv
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

load_dotenv()
sys.path.append(os.getenv("PROJECT_PATH", "."))

from data.dataset import ChestXRayDataset
from data.transforms import get_train_transforms, get_val_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositif : {device}")


# ── Dataset ────────────────────────────────────────────────────────────────

class _SplitDataset(Dataset):
    """Dataset minimal construit sur une liste (Path, label) avec transform."""

    def __init__(self, samples: list, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_counts(self) -> dict:
        c = Counter(label for _, label in self.samples)
        return {cls: c.get(i, 0) for i, cls in enumerate(ChestXRayDataset.CLASSES)}


def get_dataloaders(data_path: str, batch_size: int = 32, val_fraction: float = 0.15) -> dict:
    """Fusionne train + val officiel et crée un split stratifié 85/15.

    Le val officiel du dataset Kaggle (16 images) est trop petit pour être utile.
    Cette fonction le fusionne avec le train et recrée un split propre.
    """
    ds_train_raw = ChestXRayDataset(data_path, "train")
    ds_val_raw   = ChestXRayDataset(data_path, "val")

    all_samples = ds_train_raw.samples + ds_val_raw.samples
    all_labels  = [label for _, label in all_samples]

    idx_train, idx_val = train_test_split(
        range(len(all_samples)),
        test_size=val_fraction,
        stratify=all_labels,
        random_state=42,
    )

    ds_train = _SplitDataset([all_samples[i] for i in idx_train], get_train_transforms())
    ds_val   = _SplitDataset([all_samples[i] for i in idx_val],   get_val_transforms())
    ds_test  = ChestXRayDataset(data_path, "test", get_val_transforms())

    print(f"Split — train : {len(ds_train)} | val : {len(ds_val)} | test : {len(ds_test)}")
    print(f"  Classes train : {ds_train.get_class_counts()}")
    print(f"  Classes val   : {ds_val.get_class_counts()}")

    return {
        "train": DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=0),
        "val":   DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=0),
        "test":  DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=0),
    }


# ── Utilitaires ────────────────────────────────────────────────────────────

class EarlyStopping:
    """Early stopping sur val_acc (maximize)."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-3):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = 0.0

    def step(self, val_acc: float) -> bool:
        if val_acc > self.best + self.min_delta:
            self.best    = val_acc
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def compute_class_weights(dataset) -> torch.Tensor:
    counts = dataset.get_class_counts()
    total  = sum(counts.values())
    n      = len(ChestXRayDataset.CLASSES)
    return torch.tensor(
        [total / (n * counts[c]) for c in ChestXRayDataset.CLASSES],
        dtype=torch.float,
    )


def mixup_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2):
    """Mixup data augmentation."""
    lam     = np.random.beta(alpha, alpha)
    idx     = torch.randperm(images.size(0), device=images.device)
    mixed   = lam * images + (1 - lam) * images[idx]
    return mixed, labels, labels[idx], lam


# ── Modèle ─────────────────────────────────────────────────────────────────

def build_model(unfreeze_last_n: int = 0, dropout: float = 0.4) -> nn.Module:
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    for param in model.features.parameters():
        param.requires_grad = False

    if unfreeze_last_n > 0:
        for param in model.features[-unfreeze_last_n:].parameters():
            param.requires_grad = True

    model.classifier[0] = nn.Dropout(p=dropout)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(ChestXRayDataset.CLASSES))
    return model


# ── Boucle d'entraînement ──────────────────────────────────────────────────

def train_phase(
    model: nn.Module,
    dataloaders: dict,
    criterion: nn.Module,
    optimizer,
    scheduler,
    num_epochs: int,
    patience: int,
    save_path: str,
    run_name: str,
    use_mixup: bool = False,
) -> float:
    """Entraîne le modèle sur une phase, sauvegarde le meilleur val_acc."""
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    early_stopping = EarlyStopping(patience=patience)
    best_val_acc   = 0.0

    with mlflow.start_run(run_name=run_name, nested=True):
        for epoch in range(num_epochs):

            # — Entraînement —
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for images, labels in dataloaders["train"]:
                images, labels = images.to(device), labels.to(device)

                if use_mixup:
                    images, labels_a, labels_b, lam = mixup_batch(images, labels)
                    logits  = model(images)
                    loss    = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
                    correct = (lam * (logits.argmax(1) == labels_a).float()
                               + (1 - lam) * (logits.argmax(1) == labels_b).float()).sum().item()
                else:
                    logits  = model(images)
                    loss    = criterion(logits, labels)
                    correct = (logits.argmax(1) == labels).sum().item()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                train_loss    += loss.item() * images.size(0)
                train_correct += correct
                train_total   += images.size(0)

            train_loss /= train_total
            train_acc   = train_correct / train_total

            # — Validation —
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0

            with torch.no_grad():
                for images, labels in dataloaders["val"]:
                    images, labels = images.to(device), labels.to(device)
                    logits       = model(images)
                    val_loss    += criterion(logits, labels).item() * images.size(0)
                    val_correct += (logits.argmax(1) == labels).sum().item()
                    val_total   += images.size(0)

            val_loss /= val_total
            val_acc   = val_correct / val_total
            lr_now    = optimizer.param_groups[-1]["lr"]

            mlflow.log_metrics({
                "train_loss": train_loss, "val_loss": val_loss,
                "train_acc":  train_acc,  "val_acc":  val_acc,
                "lr":         lr_now,
            }, step=epoch)

            print(
                f"  Époque {epoch + 1:>3}/{num_epochs} | "
                f"train {train_acc:.3f} ({train_loss:.4f}) | "
                f"val {val_acc:.3f} ({val_loss:.4f}) | "
                f"lr={lr_now:.2e}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc":          best_val_acc,
                    "val_loss":         val_loss,
                }, save_path)
                print(f"    → Checkpoint (val_acc={best_val_acc:.4f})")
                mlflow.log_artifact(save_path)

            if early_stopping.step(val_acc):
                print(f"\n  Early stopping à l'époque {epoch + 1}.")
                break

    print(f"\n  Meilleur val_acc : {best_val_acc:.4f}\n")
    return best_val_acc


# ── Point d'entrée ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_path = os.getenv("DATA_PATH")

    SAVE_PATH   = "models/best_model.pth"
    BATCH_SIZE  = 32

    # Phase 1 — tête seule
    LR_P1       = 1e-3
    EPOCHS_P1   = 15
    PATIENCE_P1 = 5

    # Phase 2 — fine-tuning backbone
    LR_P2       = 3e-4
    EPOCHS_P2   = 40
    PATIENCE_P2 = 8
    UNFREEZE_N  = 3

    dataloaders   = get_dataloaders(data_path, batch_size=BATCH_SIZE)
    class_weights = compute_class_weights(dataloaders["train"].dataset).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    with mlflow.start_run(run_name="efficientnet_b0_v3"):

        # ── Phase 1 : head training ────────────────────────────────────────
        print("══ Phase 1 : head training ══════════════════════════════════════")
        model = build_model(unfreeze_last_n=0, dropout=0.4).to(device)

        opt_p1 = optim.AdamW(model.classifier.parameters(), lr=LR_P1, weight_decay=1e-4)
        sch_p1 = optim.lr_scheduler.OneCycleLR(
            opt_p1, max_lr=LR_P1,
            steps_per_epoch=len(dataloaders["train"]),
            epochs=EPOCHS_P1, pct_start=0.1,
        )

        train_phase(
            model, dataloaders, criterion, opt_p1, sch_p1,
            num_epochs=EPOCHS_P1, patience=PATIENCE_P1,
            save_path=SAVE_PATH, run_name="phase1_head",
            use_mixup=False,
        )

        # ── Phase 2 : fine-tuning ──────────────────────────────────────────
        print("══ Phase 2 : fine-tuning ════════════════════════════════════════")
        ckpt = torch.load(SAVE_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Poids phase 1 chargés (val_acc={ckpt['val_acc']:.4f})")

        for param in model.features[-UNFREEZE_N:].parameters():
            param.requires_grad = True

        opt_p2 = optim.AdamW([
            {"params": model.features[-UNFREEZE_N:].parameters(), "lr": LR_P2 / 10},
            {"params": model.classifier.parameters(),             "lr": LR_P2},
        ], weight_decay=1e-4)

        sch_p2 = optim.lr_scheduler.OneCycleLR(
            opt_p2, max_lr=[LR_P2 / 10, LR_P2],
            steps_per_epoch=len(dataloaders["train"]),
            epochs=EPOCHS_P2, pct_start=0.05,
        )

        train_phase(
            model, dataloaders, criterion, opt_p2, sch_p2,
            num_epochs=EPOCHS_P2, patience=PATIENCE_P2,
            save_path=SAVE_PATH, run_name="phase2_finetune",
            use_mixup=True,
        )

    print(f"Entraînement terminé. Modèle sauvegardé dans {SAVE_PATH}")
