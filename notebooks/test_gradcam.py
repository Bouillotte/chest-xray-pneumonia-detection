import numpy as np

import os
import sys
from dotenv import load_dotenv
load_dotenv()
project_path = os.getenv("PROJECT_PATH")
data_path = os.getenv("DATA_PATH")
sys.path.append(project_path)

from src.models.train import get_model
from src.utils.gradcam import generate_gradcam, save_gradcam

model = get_model() #on charge le modèle
model.eval() #on enlève le dropout pour ne pas perturber les calculs

from src.data.dataset import ChestXRayDataset
from src.data.transforms import get_val_transforms

image = ChestXRayDataset(data_path, "test", get_val_transforms())
image_tensor, label = image.__getitem__(0)

image_tensor = image_tensor.unsqueeze(0) #on met sous un format batch pour gradcam

#transformation de l'image normale en np.array pour la plot
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image_original = image_tensor.squeeze().permute(1,2,0).numpy()
image_original = std * image_original + mean
image_original = np.clip(image_original, 0, 1)

viz = generate_gradcam(model, image_tensor, image_original)
save_gradcam(viz, image_original, label_pred=0, label_true=label, save_path="gradcam_test.png")
print("Sauvegardé !")