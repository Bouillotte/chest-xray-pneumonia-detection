# Rapport technique — Classification de radiographies thoraciques par apprentissage profond

---

## 1. Présentation du problème

### 1.1 Contexte

La pneumonie est l'une des principales causes de mortalité infantile dans le monde. Son diagnostic repose en grande partie sur l'analyse de radiographies thoraciques (chest X-rays), une tâche qui requiert une expertise médicale avancée et qui présente une variabilité inter-observateurs non négligeable. L'automatisation partielle de ce diagnostic, ou du moins l'assistance à sa réalisation, constitue un cas d'usage concret pour l'apprentissage profond en imagerie médicale.

### 1.2 Formulation de la tâche

Il s'agit d'un problème de **classification supervisée multi-classe** :

- **Entrée** : image de radiographie thoracique (format JPEG, résolution variable)
- **Sortie** : probabilités d'appartenance à chacune des 3 classes
  - `NORMAL` — poumon sain
  - `BACTERIA` — pneumonie bactérienne (consolidations lobaires)
  - `VIRUS` — pneumonie virale (infiltrats bilatéraux diffus)

La distinction entre pneumonie bactérienne et virale est cliniquement importante car elle oriente le traitement : les pneumonies bactériennes répondent aux antibiotiques, les virales non.

---

## 2. Données

### 2.1 Dataset

Le dataset utilisé est [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) disponible sur Kaggle, constitué de **5 856 radiographies** issues de l'hôpital Guangzhou Women and Children's Medical Center.

Distribution brute :

| Partition | NORMAL | PNEUMONIA (total) | Total |
|---|---|---|---|
| train | 1 341 | 3 875 | 5 216 |
| val | 8 | 8 | 16 |
| test | 234 | 390 | 624 |

La partition de pneumonie n'est pas subdivisée en dossiers bactérie/virus. La classe est inférée depuis le **nom du fichier** : les fichiers contenant `"bacteria"` sont labellisés BACTERIA, les autres VIRUS.

### 2.2 Déséquilibre de classe

Le dataset est fortement déséquilibré : environ **3× plus de cas de pneumonie** que de cas normaux dans l'entraînement. Sans correction, un modèle naïf maximise son accuracy en prédisant systématiquement "PNEUMONIA", ce qui donne de bons chiffres globaux mais une sensibilité nulle sur NORMAL.

Correction appliquée : **pondération inverse des classes** dans la fonction de perte.

```
poids[c] = N_total / (nb_classes × N_c)
```

Cela revient à sur-pondérer les erreurs sur les classes rares et sous-pondérer celles sur les classes fréquentes.

### 2.3 Problème du split de validation officiel

La partition de validation officielle du dataset ne contient que **16 images** (8 par classe). Un tel échantillon est statistiquement insuffisant : l'accuracy calculée sur 16 images a une variance énorme, ce qui rend le suivi de l'entraînement instable et les comparaisons de configurations dénuées de sens.

**Solution** : fusion des partitions `train` et `val` (5 232 images), puis recréation d'un split **stratifié 85/15** via `sklearn.model_selection.train_test_split`. Le split stratifié maintient les mêmes proportions de chaque classe dans train et val, ce qui garantit que le signal de validation est représentatif de la distribution réelle.

Résultat du split :
- Entraînement : ~4 447 images
- Validation : ~785 images
- Test : 624 images (partition officielle Kaggle, **non touchée pendant l'entraînement**)

---

## 3. Prétraitement et augmentation

### 3.1 Principe général

Les radiographies présentent des variations naturelles de cadrage, d'exposition et de positionnement patient. L'augmentation artificielle de données simule ces variations pour rendre le modèle plus robuste et réduire l'overfitting.

Une règle fondamentale a guidé le choix des augmentations : **ne pas appliquer de transformations anatomiquement absurdes**. Par exemple, un flip vertical (retourner l'image tête en bas) inverse le thorax et l'abdomen — ce qui n'existe pas en pratique — et introduit des exemples que le modèle ne rencontrera jamais en production.

### 3.2 Pipeline d'augmentation (entraînement uniquement)

| Transformation | Paramètres | Justification |
|---|---|---|
| Resize(256) + RandomCrop(224) | — | Simule les différences de cadrage et de distance d'acquisition |
| RandomHorizontalFlip | p=0.5 | Valide anatomiquement : les poumons sont symétriques gauche/droite |
| RandomRotation | ±15° | Simule les légères inclinaisons du patient lors de la prise de vue |
| RandomAffine | translate 5%, shear 5° | Simule les variations de positionnement |
| ColorJitter | brightness 0.3, contrast 0.3 | Simule les variations de réglage du capteur Rx |
| GaussianBlur | σ ∈ [0.1, 1.0] | Simule les différences de netteté entre appareils |
| RandomErasing | p=0.2, zone 2-10% | Simule des occultations partielles ; force le modèle à ne pas s'appuyer sur une seule zone |

### 3.3 Pipeline de validation / test / inférence

Sans augmentation : `Resize(224) → ToTensor → Normalize(ImageNet)`.

La normalisation utilise les **statistiques ImageNet** (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). Bien que les radiographies soient des images médicales en niveaux de gris convertis en RGB, ces valeurs restent appropriées car le modèle a été pré-entraîné sur ImageNet avec ces statistiques.

---

## 4. Architecture du modèle

### 4.1 Choix du backbone : EfficientNet-B0

**EfficientNet-B0** (Tan & Le, 2019) a été retenu parmi les architectures disponibles dans torchvision pour plusieurs raisons :

- **Efficacité** : 5,3 M de paramètres, parmi les plus légers des architectures modernes, ce qui réduit le risque d'overfitting sur un dataset de taille modeste
- **Compound scaling** : EfficientNet équilibre largeur, profondeur et résolution du réseau, ce qui donne de meilleures features multi-échelles — utiles pour des structures anatomiques de tailles variées
- **Transfert learning** : pré-entraîné sur ImageNet (1,2 M d'images, 1 000 classes), il dispose de représentations génériques (bords, textures, formes) réutilisables pour les radiographies

La tête de classification originale (1 000 classes) est remplacée par une couche linéaire **3 classes** :

```
Dropout(p=0.4) → Linear(1280, 3)
```

Le Dropout est augmenté de 0.2 à **0.4** pour renforcer la régularisation sur ce dataset relativement petit.

### 4.2 Transfer learning

L'idée centrale du transfer learning est qu'un réseau entraîné à distinguer des milliers d'objets visuels a développé des détecteurs de features utiles (contours, textures, formes) dans ses premières couches. Ces features sont génériques et réutilisables — même pour des radiographies médicales. En repartant de ces poids plutôt que de zéro, on converge plus vite et on obtient de meilleures performances avec moins de données.

---

## 5. Stratégie d'entraînement

### 5.1 Deux phases

L'entraînement est découpé en deux phases séquentielles pour éviter un problème classique du fine-tuning : si on dégèle tout le réseau dès le début avec un LR inadapté, les gradients importants de la tête non initialisée "contaminent" le backbone et détruisent les features apprises sur ImageNet.

**Phase 1 — Head training** (backbone gelé)

- Tous les paramètres du backbone ont `requires_grad = False`
- Seule la tête de classification est entraînée
- Durée : 15 époques maximum, early stopping patience=5
- Objectif : initialiser correctement la tête avant tout fine-tuning

**Phase 2 — Fine-tuning** (backbone partiellement dégelé)

- Les 3 derniers blocs convolutifs sur 7 sont dégelés (`features[-3:]`)
- Le backbone complet ne sera jamais dégelé : les premières couches (détecteurs de bords, textures) n'ont pas besoin d'être adaptées
- Durée : 40 époques maximum, early stopping patience=8

### 5.2 Optimiseur : AdamW

**AdamW** (Adam with decoupled Weight Decay) est utilisé pour les deux phases.

Avantages sur Adam standard :
- Le weight decay est appliqué directement sur les poids et non sur le gradient normalisé, ce qui donne une régularisation plus propre
- Meilleure généralisation empiriquement démontrée sur de nombreuses tâches de vision

Paramètres :
- Weight decay = 1e-4
- Phase 1 : LR = 1e-3 (tête uniquement)
- Phase 2 : LR différentiel — backbone dégelé = 3e-5, tête = 3e-4

Le **LR différentiel** en phase 2 est une technique clé : le backbone a déjà de bonnes features, il ne faut le modifier que très légèrement (LR/10). La tête, en revanche, peut être modifiée plus agressivement.

### 5.3 Scheduler : OneCycleLR

Le scheduler **OneCycleLR** suit une politique en trois temps :
1. **Warmup** (pct_start % des steps) : montée progressive du LR de LR_min vers max_lr
2. **Cosine annealing** : descente en cosinus de max_lr vers un LR très faible
3. Fin : LR au plus bas pour une convergence stable

Avantages sur ReduceLROnPlateau :
- Le warmup initial stabilise l'entraînement dans les premières époques
- Le scheduling est déterministe (pas de dépendance aux métriques de validation)
- Meilleure exploration de l'espace de paramètres avant de converger

### 5.4 Early stopping

L'early stopping surveille `val_acc` (et non `val_loss`). Sur des données déséquilibrées, la val_loss peut continuer à baisser même si le modèle commence à se spécialiser sur la classe majoritaire — ce qui ne se reflète pas dans l'accuracy.

Si val_acc ne progresse pas d'au moins 0.001 pendant `patience` époques consécutives, l'entraînement s'arrête. Le meilleur checkpoint (au sens de val_acc) est sauvegardé.

---

## 6. Régularisation avancée

### 6.1 Mixup (phase 2 uniquement)

Mixup (Zhang et al., 2018) est une technique de régularisation qui crée des exemples synthétiques en interpolant linéairement deux images et leurs labels :

```
image_mixup = λ × image_A + (1 - λ) × image_B
loss = λ × loss(prédiction, label_A) + (1 - λ) × loss(prédiction, label_B)
```

λ est tiré d'une distribution Beta(α, α) avec α = 0.2.

Effets :
- Lisse la frontière de décision entre classes
- Réduit l'overconfidence du modèle
- Force le modèle à interpoler entre classes plutôt que mémoriser des exemples

Mixup n'est activé qu'en phase 2 (fine-tuning). En phase 1, la tête est initialisée depuis zéro — ajouter du Mixup ralentirait inutilement cette initialisation.

### 6.2 Label smoothing (ε = 0.1)

Le label smoothing modifie les cibles de la cross-entropy :
- Au lieu de [0, 0, 1] pour VIRUS, la cible devient [0.05, 0.05, 0.90]

Cela empêche le modèle d'être trop certain de ses prédictions et améliore la calibration des probabilités.

### 6.3 Gradient clipping

La norme des gradients est clippée à 1.0 à chaque step :

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Cela prévient les explosions de gradient qui peuvent survenir lors du fine-tuning, notamment en début de phase 2 quand le backbone vient d'être dégelé.

---

## 7. Inférence — Test Time Augmentation (TTA)

À l'inférence, on applique une **Test Time Augmentation** légère : l'image originale et son flip horizontal sont tous deux passés dans le modèle, et les probabilités softmax sont moyennées.

Formellement, pour une image I :

```
p_finale = (softmax(f(I)) + softmax(f(flip(I)))) / 2
```

Le flip horizontal est anatomiquement valide car les deux poumons présentent des caractéristiques radiographiques symétriques. Cette technique améliore légèrement la robustesse des prédictions sans nécessiter de réentraînement.

---

## 8. Interprétabilité — Grad-CAM

**Grad-CAM** (Selvaraju et al., 2017) génère une carte de chaleur indiquant quelles zones de l'image ont le plus influencé la décision du modèle.

Principe : les gradients de la score de la classe prédite par rapport aux feature maps de la dernière couche convolutive (`features[-1]`) sont calculés. Ces gradients sont moyennés spatialement pour obtenir des poids par canal, puis pondèrent les feature maps. Le résultat est une carte 2D interpolée à la taille de l'image d'entrée.

La carte est superposée à la radiographie originale avec un colormap rouge-bleu : les zones rouges correspondent aux régions déterminantes pour la prédiction.

**Limitation importante** : Grad-CAM montre les patterns que le modèle a appris à associer à chaque classe — pas nécessairement les zones cliniquement pertinentes. Des biais dans le dataset (taille de la cage thoracique corrélée à l'âge, format des prises de vue) peuvent amener le modèle à activer des zones non pulmonaires. Pour un outil clinique réel, une supervision anatomique serait nécessaire.

---

## 9. Résultats et analyse

### 9.1 Métriques obtenues

Après entraînement (Phase 1 + Phase 2) :

| Métrique | Valeur |
|---|---|
| Meilleur val_acc (Phase 2) | ~73.9 % |

### 9.2 Analyse des résultats

Une val_acc de 73.9 % sur 3 classes (baseline aléatoire : 33 %) montre que le modèle a bien appris à discriminer les classes. Sur un problème médical à 3 classes avec ~785 images de validation, ce résultat est cohérent avec l'état de l'art sur ce dataset pour une architecture légère.

Les principales sources d'erreur attendues sont :
- La confusion **BACTERIA / VIRUS** : les deux formes de pneumonie peuvent présenter des aspects radiographiques proches, et même les radiologues humains ne les distinguent pas toujours avec certitude
- Les cas **NORMAL mal classés comme BACTERIA** : les faux positifs de pneumonie sont un problème cliniquement important, plus grave que les faux négatifs dans certains contextes

### 9.3 Limites du modèle

1. **Corrélations parasites** : le modèle n'a pas de supervision anatomique. Il peut exploiter des biais visuels du dataset (position du patient, format de l'image) plutôt que des caractéristiques purement pulmonaires.

2. **Dataset non représentatif** : toutes les images proviennent d'un seul hôpital pédiatrique (Guangzhou). Les performances sur des images d'adultes ou d'autres appareils radiographiques ne sont pas garanties.

3. **Calibration** : bien que le label smoothing et le Mixup améliorent la calibration, les probabilités softmax ne sont pas directement interprétables comme des probabilités cliniques. Une calibration post-hoc (Platt scaling, temperature scaling) serait nécessaire pour un usage médical.

4. **Absence de segmentation pulmonaire** : sans masque des poumons, Grad-CAM peut activer des zones non pertinentes, limitant la valeur interprétative de la visualisation.

---

## 10. Conclusion

Ce projet implémente une pipeline complète de classification de radiographies thoraciques, depuis la préparation des données jusqu'au déploiement d'une interface interactive. Les principales contributions techniques sont :

- Un split de validation reconstruit (résout le problème des 16 images officielles)
- Une stratégie d'entraînement en deux phases avec LR différentiel
- Une combinaison de techniques de régularisation complémentaires (Dropout, Mixup, label smoothing, RandomErasing)
- Un scheduler OneCycleLR pour une convergence stable
- Une pondération des classes pour corriger le déséquilibre du dataset
- Un TTA léger à l'inférence
- Une visualisation Grad-CAM avec discussion honnête de ses limites

Le modèle obtient ~73.9 % d'accuracy sur le set de validation reconstruit. Les limitations identifiées (biais dataset, absence de segmentation, dataset mono-source) constituent des pistes d'amélioration claires pour une version future.
