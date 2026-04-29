"""Chargement du jeu de données de radiographies pulmonaires.

Structure attendue sur le disque::

    <data_dir>/
        train/
            NORMAL/    (fichiers *.jpeg)
            PNEUMONIA/ (fichiers *.jpeg)
        val/  (idem)
        test/ (idem)

Les fichiers du dossier ``PNEUMONIA`` dont le nom contient ``"bacteria"`` sont assignés
au label 1 (pneumonie bactérienne) ; les autres au label 2 (pneumonie virale).
"""

from collections import Counter
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ChestXRayDataset(Dataset):
    """Jeu de données de radiographies thoraciques avec 3 classes.

    Correspondance des labels :
        0 → NORMAL
        1 → BACTERIA (pneumonie bactérienne)
        2 → VIRUS    (pneumonie virale)
    """

    CLASSES = ["NORMAL", "BACTERIA", "VIRUS"]

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        transform=None,
    ) -> None:
        """
        Args:
            data_dir:  Répertoire racine du jeu de données.
            split:     Partition à charger (``"train"``, ``"val"`` ou ``"test"``).
            transform: Pipeline de transformations torchvision appliqué à chaque image.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self._charger_echantillons()

    def _charger_echantillons(self) -> None:
        """Parcourt le répertoire de la partition et construit la liste (chemin, label)."""
        rep_normal = self.data_dir / self.split / "NORMAL"
        for chemin in sorted(rep_normal.glob("*.jpeg")):
            self.samples.append((chemin, 0))

        rep_pneumonie = self.data_dir / self.split / "PNEUMONIA"
        for chemin in sorted(rep_pneumonie.glob("*.jpeg")):
            label = 1 if "bacteria" in chemin.stem else 2
            self.samples.append((chemin, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Charge et retourne ``(image, label)`` pour l'indice ``idx``.

        Args:
            idx: Indice de l'échantillon dans la liste.

        Returns:
            Tuple ``(image, label)`` où ``image`` est un tenseur si ``transform`` est
            défini, ou une ``PIL.Image`` sinon.
        """
        chemin, label = self.samples[idx]
        image = Image.open(chemin).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def get_class_counts(self) -> dict[str, int]:
        """Retourne le nombre d'images par classe.

        Returns:
            Dictionnaire ``{nom_classe: nombre_images}``.

        >>> from pathlib import Path
        >>> ds = ChestXRayDataset.__new__(ChestXRayDataset)
        >>> ds.samples = [(Path("a.jpeg"), 0), (Path("b.jpeg"), 1), (Path("c.jpeg"), 0)]
        >>> ds.get_class_counts()
        {'NORMAL': 2, 'BACTERIA': 1, 'VIRUS': 0}
        """
        compteur = Counter(label for _, label in self.samples)
        return {cls: compteur.get(i, 0) for i, cls in enumerate(self.CLASSES)}
