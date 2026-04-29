"""Pipelines de transformations torchvision pour l'entraînement et la validation."""

from torchvision import transforms

# Statistiques de normalisation ImageNet (RGB) — communes à l'entraînement et à l'inférence
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224


def get_train_transforms() -> transforms.Compose:
    """Retourne le pipeline d'augmentation pour la phase d'entraînement.

    Applique des transformations aléatoires (flip, rotation, luminosité/contraste)
    afin de réduire le surapprentissage, puis normalise selon les statistiques ImageNet.

    Returns:
        Pipeline ``transforms.Compose`` prêt à l'emploi.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """Retourne le pipeline de transformation pour la validation et le test.

    Aucune augmentation : redimensionnement et normalisation ImageNet uniquement.

    Returns:
        Pipeline ``transforms.Compose`` prêt à l'emploi.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
