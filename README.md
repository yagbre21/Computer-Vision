# Image Classification of Wild Edible Plants Using Transfer Learning with EfficientNetB0 and MobileNet

**CSCI E-25: Computer Vision - Spring 2024**
Harvard University, Division of Continuing Education
Instructor: Dr. Stephen Elston, PhD

**Author:** Yves Agbre

## Abstract

This project develops an image classification system for identifying 35 species of wild edible plants from photographs. Given the safety-critical nature of the task - misidentification of wild plants can result in serious illness or death - the system prioritizes classification accuracy. Two convolutional neural network architectures, EfficientNetB0 and MobileNet, are evaluated using transfer learning from ImageNet. EfficientNetB0 achieves 89.7% test accuracy with 89.9% precision and 89.7% recall, significantly outperforming MobileNet (79.2% accuracy) on the same dataset and training pipeline. Supplementary fine-tuning of EfficientNetB0 further improved accuracy to 91.7%.

## 1. Introduction

Foraging for wild edible plants has grown in popularity, but the risk of misidentification remains significant. Automated visual classification systems can serve as a decision support tool for identifying edible species in the field. This project compares two lightweight CNN architectures suited for mobile deployment, evaluating their accuracy-efficiency tradeoffs on a multi-class plant identification task.

## 2. Dataset

- **Source:** Kaggle Wild Edible Plants dataset
- **Total images:** 16,526
- **Classes:** 35 edible plant species
- **Image dimensions:** 224 x 224 x 3 (RGB)
- **Split:** 70% training (11,568) / 20% validation (3,305) / 10% test (1,653)

### Species

Alfalfa, Allium, Borage, Burdock, Calendula, Cattail, Chickweed, Chicory, Chive Blossom, Coltsfoot, Common Mallow, Common Milkweed, Common Vetch, Common Yarrow, Coneflower, Cow Parsley, Cowslip, Crimson Clover, Crithmum Maritimum, Daisy, Dandelion, Fennel, Fireweed, Gardenia, Garlic Mustard, Geranium, Ground Ivy, Harebell, Henbit, Knapweed, Meadowsweet, Mullein, Pickerelweed, Ramsons, Red Clover

## 3. Methodology

### 3.1 Preprocessing
- Images resized to 224x224 pixels
- Pixel normalization via architecture-specific preprocessing functions
- Labels one-hot encoded for 35-class classification

### 3.2 Data Augmentation
To improve generalization across varying field conditions, the following augmentations were applied during training:
- Random rotation, zoom, and horizontal flip
- Brightness and contrast adjustment
- Scaling transformations

### 3.3 Model Architectures

Both models use transfer learning from ImageNet pre-trained weights with custom classification heads.

**EfficientNetB0:** Compound-scaled architecture that uniformly scales depth, width, and resolution. The base model's convolutional layers were used as a feature extractor, with custom dense layers and a 35-class softmax output.

**MobileNet:** Depthwise separable convolution architecture optimized for mobile and edge deployment. Same custom classification head structure as EfficientNetB0.

### 3.4 Training Configuration
- **Runtime:** Google Colab with TPU (v28) acceleration
- **Optimizer:** Adam (with weight decay for transfer learning phases)
- **Loss:** Categorical cross-entropy
- **Metrics:** Accuracy, Precision, Recall (via sklearn)
- **Callbacks:** Early stopping, model checkpointing (best validation loss)
- **EfficientNetB0:** 35 epochs transfer learning, plus supplementary fine-tuning (top 20 layers unfrozen)
- **MobileNet:** 35 epochs with Keras precision and recall tracking

## 4. Results

### Transfer Learning (Primary Evaluation)

| Model | Test Accuracy | Precision | Recall |
|-------|:------------:|:---------:|:------:|
| **EfficientNetB0** | **89.7%** | **89.9%** | **89.7%** |
| MobileNet | 79.2% | 80.6% | 78.2% |

EfficientNetB0 outperformed MobileNet by 10.5 percentage points in test accuracy, with comparable gains in precision and recall. Both models used transfer learning from ImageNet pre-trained weights with custom classification heads trained on the plant dataset.

MobileNet showed signs of overfitting in later epochs, with training accuracy reaching 98.8% while validation accuracy plateaued around 83.5%.

### Supplementary Fine-Tuning

As an additional experiment, the top 20 layers of EfficientNetB0 were unfrozen (excluding BatchNorm layers) and retrained with a reduced learning rate (1e-5). This fine-tuning phase improved test accuracy to **91.7%** (test loss: 0.288), demonstrating the benefit of allowing deeper feature adaptation to the plant domain.

## 5. Conclusion

EfficientNetB0 demonstrates strong performance on multi-class plant identification, achieving 89.7% accuracy via transfer learning and 91.7% after supplementary fine-tuning across 35 species. The compound scaling approach provides meaningful accuracy gains over MobileNet's depthwise separable architecture for this task. Both models benefit substantially from ImageNet transfer learning, converging to useful accuracy levels within the first 10 epochs.

For deployment in a field-use application, EfficientNetB0 offers the better accuracy-to-compute ratio for this classification task. Future work could explore fine-grained augmentation strategies, ensemble methods, or larger EfficientNet variants (B1-B7) to push accuracy closer to the threshold needed for safety-critical deployment.

## Repository Structure

```
Plant_Classification_Harvard_CV_Project.ipynb    # Full notebook: EDA, augmentation, training, evaluation
Deep-Learning-With-Python/                       # Supplementary deep learning exercises
```

## Requirements

- Python 3.x
- TensorFlow 2.15+
- NumPy, Matplotlib, Pandas, Seaborn
- Kaggle API (for dataset download)

## Course Reference

- [CSCI E-25 Course Materials (Prof. Elston)](https://github.com/StephenElston/CSCI-E25)
- [Course Listing - Harvard DCE](https://coursebrowser.dce.harvard.edu/course/computer-vision/)

## Author

**Yves Agbre** - Harvard ALM, Management | [LinkedIn](https://www.linkedin.com/in/yagbre)
