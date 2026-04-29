---
title: Chest X-Ray Pneumonia Detection
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
tags:
- streamlit
- pytorch
- medical-imaging
- deep-learning
pinned: false
short_description: Pneumonia detection from chest X-rays
license: mit
---

# Détection de pneumonie par radiographie thoracique

Classificateur de radiographies pulmonaires en 3 classes — **Normal**, **Pneumonie bactérienne**, **Pneumonie virale** — basé sur EfficientNet-B0 fine-tuné avec PyTorch.

L'interface propose une visualisation Grad-CAM pour identifier les zones de l'image déterminantes pour la prédiction.

---

## Aperçu

| Classe | Description |
|---|---|
| NORMAL | Poumon sain |
| BACTERIA | Pneumonie bactérienne |
| VIRUS | Pneumonie virale |

---

## Architecture

- **Backbone** : EfficientNet-B0 pré-entraîné ImageNet
- **Entraînement** : gel du feature extractor, puis fine-tuning des 3 derniers blocs
- **Entrée** : image 224×224, normalisée ImageNet
- **Sortie** : probabilités sur 3 classes (softmax)
- **Interprétabilité** : Grad-CAM sur la dernière couche convolutive

---

## Lancer l'application

### Avec Docker

```bash
docker build -t pneumonia-detection .
docker run -p 8501:8501 pneumonia-detection
```

Accès : [http://localhost:8501](http://localhost:8501)

### En local

```bash
pip install -r requirements.txt
streamlit run app.py
```

### API REST (FastAPI)

```bash
uvicorn api.main:app --reload
```

Routes disponibles :

| Méthode | Route | Description |
|---|---|---|
| GET | `/` | Statut de l'API |
| GET | `/health` | État du modèle chargé |
| POST | `/predict` | Prédiction sur une image |

Exemple d'appel :

```bash
curl -X POST http://localhost:8000/predict \
  -F "fichier=@radio.jpg"
```

---

## Structure du projet

```
.
├── api/                  # API REST FastAPI
├── app/                  # Inférence et app Streamlit locale
├── data/                 # Dataset et transformations
├── models/               # Entraînement, évaluation, checkpoint
├── notebooks/            # Scripts d'exploration et tests
├── utils/                # Grad-CAM
├── app.py                # Point d'entrée Streamlit (Docker / HuggingFace)
├── Dockerfile
└── requirements.txt
```

---

## Dataset

Le modèle a été entraîné sur le dataset [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) de Kaggle.

Structure attendue :

```
<DATA_PATH>/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/   # fichiers contenant "bacteria" ou "virus" dans le nom
├── val/
└── test/
```

---

## Entraînement

Configurer les variables d'environnement dans un fichier `.env` :

```
PROJECT_PATH=/chemin/vers/le/projet
DATA_PATH=/chemin/vers/le/dataset
```

Lancer l'entraînement :

```bash
python models/train.py
```

Les métriques sont enregistrées via MLflow. Un checkpoint est sauvegardé à chaque nouveau minimum de validation loss.

---

## Avertissement

Cet outil est un prototype de recherche. Il ne remplace pas un diagnostic médical établi par un professionnel de santé qualifié.
