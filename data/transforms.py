"""Pipelines de transformations torchvision pour l'entraînement et la validation."""

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_SIZE    = 224
IMAGE_SIZE_LG = 256


def get_train_transforms() -> transforms.Compose:
    """Pipeline d'augmentation pour l'entraînement.

    Stratégie adaptée aux radiographies pulmonaires :
    - Resize vers 256 + RandomCrop 224 : diversité de cadrage
    - RandomHorizontalFlip : symétrie gauche/droite anatomiquement valide
    - PAS de flip vertical (inverser thorax/abdomen n'a pas de sens)
    - RandomAffine : simule les variations de positionnement patient
    - ColorJitter : simule les variations de réglage de l'appareil Rx
    - GaussianBlur : simule les différences de qualité d'acquisition
    - RandomErasing : force le modèle à s'appuyer sur des zones distribuées
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE_LG, IMAGE_SIZE_LG)),
        transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])


def get_val_transforms() -> transforms.Compose:
    """Pipeline de transformation pour la validation et le test (sans augmentation)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
