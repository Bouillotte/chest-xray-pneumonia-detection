# Vue d'ensemble du projet — Détection de pneumonie par radiographie thoracique

---

## 1. Objectif

Construire un classifieur automatique de radiographies pulmonaires capable de distinguer trois situations cliniques :

- **NORMAL** : poumon sain
- **BACTERIA** : pneumonie bactérienne
- **VIRUS** : pneumonie virale

Le projet couvre l'intégralité de la chaîne : préparation des données → entraînement → évaluation → inférence → interface utilisateur → déploiement.

---

## 2. Structure de l'arbre

```
chest-xray-pneumonia-detection/
│
├── data/
│   ├── dataset.py          Classe PyTorch Dataset — chargement et labellisation des images
│   └── transforms.py       Pipelines de transformations (augmentation train / val)
│
├── models/
│   ├── train.py            Script d'entraînement complet (deux phases)
│   ├── evaluate.py         Métriques test : accuracy, matrice de confusion, courbe ROC
│   └── best_model.pth      Checkpoint du meilleur modèle (suivi par Git LFS)
│
├── app/
│   ├── inference.py        Chargement du modèle + prédiction (avec TTA)
│   └── streamlit_app.py    App Streamlit en exécution locale (lit PROJECT_PATH depuis .env)
│
├── api/
│   └── main.py             API REST FastAPI — route POST /predict
│
├── utils/
│   └── gradcam.py          Génération de la carte Grad-CAM (pytorch-grad-cam)
│
├── samples/                Images d'exemple (une par classe, suivies par Git LFS)
│
├── app.py                  Point d'entrée Streamlit pour Docker / HuggingFace Spaces
├── Dockerfile              Image Docker pour le déploiement
├── requirements.txt        Dépendances Python
├── .env                    Variables d'environnement locales (non versionné)
└── README.md               Documentation publique du projet
```

---

## 3. Flux de données — de l'image brute à la prédiction

### 3.1 Chargement des données (`data/dataset.py`)

`ChestXRayDataset` est une classe héritant de `torch.utils.data.Dataset`. Elle parcourt le dossier du dataset Kaggle et construit une liste de tuples `(chemin_image, label)`.

La labellisation se fait à partir de la **structure de dossiers et des noms de fichiers** :
- `NORMAL/` → label 0
- `PNEUMONIA/` contenant `"bacteria"` dans le nom → label 1
- `PNEUMONIA/` sans `"bacteria"` → label 2 (virus)

Le dataset Kaggle ne sépare pas bactéries et virus dans des sous-dossiers distincts — il faut donc inférer la classe depuis le nom du fichier.

### 3.2 Transformations (`data/transforms.py`)

Deux pipelines distincts :

**Entraînement** (`get_train_transforms`) :
```
Resize(256) → RandomCrop(224) → RandomHorizontalFlip → RandomRotation(15°)
→ RandomAffine(translate, shear) → ColorJitter → GaussianBlur
→ ToTensor → Normalize(ImageNet) → RandomErasing
```

**Validation / Test / Inférence** (`get_val_transforms`) :
```
Resize(224) → ToTensor → Normalize(ImageNet)
```

Le pipeline de validation est volontairement sans augmentation pour mesurer les vraies performances du modèle sur des images non perturbées.

### 3.3 Split train / val (`models/train.py` — `get_dataloaders`)

Le split officiel de Kaggle comporte **seulement 16 images en validation** (8 NORMAL, 8 PNEUMONIA). Ce signal est trop faible pour guider l'entraînement ou détecter l'overfitting.

Solution : fusionner `train/` et `val/` (5 232 images au total), puis créer un split **stratifié 85/15** avec `sklearn.model_selection.train_test_split`. Le split stratifié garantit que les proportions de chaque classe sont identiques dans train et val.

Résultat : ~4 447 images d'entraînement, ~785 de validation, 624 de test (set officiel Kaggle, non touché pendant l'entraînement).

---

## 4. Modèle

### 4.1 Architecture (`models/train.py` — `build_model`)

Base : **EfficientNet-B0** pré-entraîné sur ImageNet (torchvision).

Modifications :
- La tête de classification originale (1000 classes ImageNet) est remplacée par une tête à **3 classes**
- Le `Dropout` passe de 0.2 (valeur EfficientNet par défaut) à **0.4** pour renforcer la régularisation

```python
model.classifier[0] = nn.Dropout(p=0.4)
model.classifier[1] = nn.Linear(1280, 3)
```

### 4.2 Entraînement en deux phases

**Pourquoi deux phases ?**
Un backbone pré-entraîné a des représentations utiles mais inadaptées aux radiographies. Si on dégèle tout d'un coup avec un LR élevé, on risque de "détruire" ces représentations avant que la tête ne soit stable. La stratégie deux phases évite ce problème.

**Phase 1 — Head training** (15 époques, patience 5)
- Backbone entièrement gelé (`requires_grad = False`)
- Seule la tête de classification est entraînée
- AdamW, LR = 1e-3, OneCycleLR
- Pas de Mixup

**Phase 2 — Fine-tuning** (40 époques, patience 8)
- Les 3 derniers blocs convolutifs sont dégelés
- LR différentiel : backbone = 3e-5, tête = 3e-4 (ratio ×10)
- AdamW + OneCycleLR, Mixup activé

---

## 5. Techniques de régularisation

| Technique | Rôle |
|---|---|
| Dropout 0.4 | Désactive aléatoirement des neurones → réduit la co-adaptation |
| Weight decay 1e-4 (AdamW) | Pénalise les poids trop grands → limite l'overfitting |
| Augmentation (RandomCrop, Flip, Affine…) | Diversifie artificiellement les exemples d'entraînement |
| RandomErasing | Masque aléatoirement une zone → force la robustesse |
| Mixup (phase 2) | Mélange deux images et leurs labels → régularise la frontière de décision |
| Label smoothing 0.1 | Adoucit les labels "durs" (0/1 → 0.05/0.95) → réduit l'overconfidence |
| Early stopping | Arrête l'entraînement si val_acc ne progresse plus → évite l'overfitting tardif |

---

## 6. Fonction de perte

`CrossEntropyLoss` avec **pondération par classe** et **label smoothing**.

La pondération compense le déséquilibre du dataset (environ 3× plus de cas de pneumonie que de cas normaux). Chaque classe reçoit un poids inversement proportionnel à sa fréquence :

```python
poids[c] = total / (nb_classes × nb_images_classe[c])
```

Cela évite que le modèle ne se contente de prédire "PNEUMONIA" pour tout maximiser son accuracy globale.

---

## 7. Inférence et TTA (`app/inference.py`)

À l'inférence, **Test Time Augmentation (TTA)** est appliqué :
1. L'image originale est passée dans le modèle → softmax → probabilités
2. L'image retournée horizontalement (flip) est passée → softmax → probabilités
3. Les deux vecteurs de probabilités sont **moyennés**

Le retournement horizontal est anatomiquement valide sur des radiographies thoraciques (le poumon gauche et le poumon droit ont des caractéristiques symétriques). La moyenne réduit la variance de la prédiction sans réentraîner le modèle.

---

## 8. Interprétabilité — Grad-CAM (`utils/gradcam.py`)

Grad-CAM (*Gradient-weighted Class Activation Mapping*) identifie les zones de l'image ayant le plus contribué à la prédiction.

Principe : on calcule le gradient de la score de la classe prédite par rapport aux feature maps de la dernière couche convolutive. Les zones où ce gradient est fort sont celles que le modèle "regarde" pour sa décision.

La carte résultante est superposée à l'image originale avec un colormap chaud (rouge = forte activation, bleu = faible activation).

**Limitation** : Grad-CAM n'est pas une carte anatomique — il montre les corrélations que le modèle a apprises. Si le dataset présente des biais (taille de la cage thoracique, position du patient), Grad-CAM peut activer des zones non cliniquement pertinentes.

---

## 9. Interface utilisateur (`app.py`)

Application Streamlit organisée en trois onglets :

| Onglet | Contenu |
|---|---|
| À propos | Description du projet, stack technique, avertissement médical |
| Exemples | Prédictions pré-calculées sur NORMAL.jpeg, BACTERIA.jpeg, VIRUS.jpeg |
| Démo | Upload libre d'une radiographie ou sélection d'un exemple |

Les prédictions sur les exemples sont mises en cache (`@st.cache_data`) pour éviter de recalculer à chaque interaction.

---

## 10. API REST (`api/main.py`)

FastAPI expose trois routes :

| Route | Méthode | Description |
|---|---|---|
| `/` | GET | Statut de l'API |
| `/health` | GET | Vérifie que le modèle est chargé |
| `/predict` | POST | Reçoit une image, retourne classe + confiance + probabilités |

L'API et l'interface Streamlit partagent le même module `app/inference.py` — la logique de prédiction n'est pas dupliquée.

---

## 11. Suivi des expériences — MLflow

Chaque run d'entraînement est tracé dans MLflow. Les métriques enregistrées à chaque époque :

- `train_loss`, `val_loss`
- `train_acc`, `val_acc`
- `lr` (learning rate courant)

Les runs sont organisés en runs imbriqués : un run parent `efficientnet_b0_v3` contient deux sous-runs `phase1_head` et `phase2_finetune`. Le checkpoint du meilleur modèle est logué comme artefact MLflow.

---

## 12. Déploiement

### Docker

Le `Dockerfile` crée une image autonome qui lance l'application Streamlit sur le port 8501. Le modèle `best_model.pth` est inclus dans l'image.

### HuggingFace Spaces

Le dépôt est synchronisé avec HuggingFace Spaces via `git push`. HuggingFace détecte le `Dockerfile`, construit l'image et la déploie automatiquement.

### Git LFS

Les fichiers binaires (modèle `.pth`, images `.jpeg`/`.jpg`/`.png`) sont suivis par **Git LFS** (Large File Storage). LFS stocke les binaires sur un serveur dédié et ne garde qu'un pointeur dans le dépôt Git principal — ce qui évite que le dépôt ne grossisse indéfiniment à chaque nouveau modèle.

---

## 13. Décisions importantes et pourquoi

| Décision | Raison |
|---|---|
| EfficientNet-B0 plutôt qu'un ResNet | Meilleur ratio performance/paramètres ; les radiographies bénéficient des connexions à différentes échelles |
| Split 85/15 maison plutôt que val Kaggle | Le val officiel = 16 images, signal statistiquement inutilisable |
| Deux phases d'entraînement | Évite de détruire les features ImageNet avant que la tête ne soit stable |
| LR différentiel backbone/tête | Le backbone a déjà de bonnes features, il ne faut pas le modifier trop vite |
| Sauvegarde sur val_acc plutôt que val_loss | Sur données déséquilibrées, val_loss peut baisser sans que l'accuracy réelle progresse |
| TTA flip horizontal | Anatomiquement valide, améliore la robustesse sans coût d'entraînement |
