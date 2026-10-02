# Pneumonia Diagnosis Detection and Localization (Project 34)

A deep-learning system that (1) classifies chest X-rays as **pneumonia / normal** and (2) **localizes** the suspected region with **Grad-CAM**, without using location labels during training.

> **Disclaimer:** This is an educational project. It is not a medical device and must not be used for clinical diagnosis.

## Overview

```
Chest X-ray (DICOM) -> preprocess (224x224 RGB) -> DenseNet121 -> P(pneumonia)
                                                        |
                                                    Grad-CAM -> heatmap -> thresholded region -> IoU vs ground-truth boxes
```

## Dataset

[RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) (Kaggle). Each X-ray has a binary label and, for pneumonia cases, ground-truth bounding boxes (used only for evaluating localization).

The data is **not included** in this repository. See [`data/README.md`](data/README.md) for download instructions.

- One row per X-ray, stratified split **70% train / 15% validation / 15% test**
- Final model trained on the full training split (about 18k images)
- Evaluation on a fixed 500-image test sample (106 pneumonia, 394 normal)

## Method

| Component | Details |
|---|---|
| Preprocessing | DICOM -> rescale to 0-255 -> RGB -> resize 224x224 -> ImageNet normalization |
| Model | ImageNet-pretrained DenseNet121, final layer replaced by 1 output (logit) |
| Loss | Weighted BCE (`pos_weight` = negatives / positives) for class imbalance |
| Optimizer | Adam, lr 3e-5, weight decay 1e-4, ReduceLROnPlateau |
| Augmentation | Horizontal flip, 10-degree rotation, brightness/contrast jitter |
| Stopping | Early stopping on validation AUC (patience 3); best checkpoint = epoch 3 (val AUC 0.8806) |
| Localization | Grad-CAM on the last feature map (7x7), thresholded, compared with ground-truth boxes by IoU |

A **baseline** was trained first (3,000 training images, no augmentation, lr 1e-4, 5 epochs). It overfit: validation loss rose from 0.78 to 1.23 and validation AUC peaked at epoch 1 (0.8526). The final model fixes this with more data, augmentation and regularization.

## Results (test set, threshold 0.5)

| Metric | Baseline | Final model |
|---|---|---|
| ROC-AUC | 0.852 | **0.884** |
| Accuracy | 0.772 | **0.808** |
| Precision | 0.477 | **0.531** |
| Recall (sensitivity) | 0.792 | **0.802** |
| F1-score | 0.596 | **0.639** |
| Mean IoU (Grad-CAM threshold 0.5) | 0.218 | **0.279** |

Final model: specificity 0.810; confusion matrix **TN = 319, FP = 75, FN = 21, TP = 85**.

**Localization** (106 pneumonia cases), mean IoU by Grad-CAM threshold:

| Threshold | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|
| Baseline | 0.213 | 0.222 | 0.218 | 0.203 | 0.176 |
| Final model | 0.261 | 0.274 | 0.279 | 0.267 | 0.247 |

Pointing-game hit rate (hottest heatmap pixel lies inside a true box): **47.2%**.

## Limitations

- Single train/validation/test split; no cross-validation
- Evaluated on a 500-image test sample, so estimates carry uncertainty
- Moderate precision (75 false positives on 394 normal images)
- Grad-CAM is coarse (7x7 feature map): it gives approximate regions, not precise boundaries; IoU is modest (about 0.28)
- Heatmaps sometimes highlight the heart/diaphragm rather than the lung fields, suggesting the model may use non-lung cues
- No external validation; not a clinical tool

## Repository structure

```
.
├── README.md
├── requirements.txt
├── notebooks/              # Colab notebook: data, training, evaluation, Grad-CAM, demo
├── models/
│   └── best_densenet121_v2.pth    # final weights (epoch 3)
├── results/                # metrics CSVs, confusion matrix, ROC curve, IoU plots, demo gallery
├── demo_samples/           # example X-rays (PNG) for the demo
└── data/
    └── README.md           # how to download the dataset
```

## How to run

The project was developed in **Google Colab** with a GPU (T4).

1. Open the notebook in `notebooks/` in Google Colab and set **Runtime -> Change runtime type -> GPU**.
2. Install dependencies: `pip install -r requirements.txt` (Colab already includes most of them; `pydicom` is the one usually missing).
3. Download the RSNA dataset (see `data/README.md`) and make sure the notebook's `path` variable points to the folder containing `stage_2_train_labels.csv` and `stage_2_train_images/`.
4. Run the cells top to bottom to reproduce data splits, training, evaluation and Grad-CAM/IoU.

### Run the demo without retraining

1. Put `models/best_densenet121_v2.pth` and the images from `demo_samples/` where the notebook can read them (for example in Google Drive).
2. In the notebook, run the cells that define the model, the transform and `compute_gradcam`, then the demo cell. It loads the saved weights, takes an X-ray, and shows the prediction with confidence, the Grad-CAM heatmap, and a box around the hottest region.

Demo images: `good_pneumonia_*` are detected cases with good localization (selected from the best-localized test detections, so they are **not representative** of average performance), `good_normal_*` are confidently cleared normals, and `missed_pneumonia_1` is a failure case (P(pneumonia) = 0.385).

## Reference

- RSNA Pneumonia Detection Challenge, Kaggle
- Stanford CS229 project on pneumonia diagnosis detection and localization (CNN + CAM)
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017
- Huang et al., *Densely Connected Convolutional Networks*, CVPR 2017
