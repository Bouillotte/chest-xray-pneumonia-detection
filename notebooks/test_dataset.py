from dotenv import load_dotenv
import os
import sys

load_dotenv() #on charge le env
chemin_dataset = os.getenv("DATA_PATH") #on récupère le chemin d'accès du dataset
project_path = os.getenv("PROJECT_PATH")
sys.path.append(project_path)

from src.data.dataset import ChestXRayDataset

dataset = ChestXRayDataset(chemin_dataset, split='train')
print(len(dataset))

print(dataset.get_class_counts())

from src.data.transforms import get_train_transforms

transform = get_train_transforms()
dataset_avec_transform = ChestXRayDataset(chemin_dataset, "train", transform)

image, label = dataset_avec_transform[0]
print(f"Type : {type(image)}")
print(f"Forme de tenseur : {image.shape}")
print(f"Label : {ChestXRayDataset.CLASSES[label]} ({label})")