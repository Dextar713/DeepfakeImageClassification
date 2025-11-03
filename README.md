# Deepfake Image Classification

A deep learning project for classifying deepfake images into 5 categories using convolutional neural networks and ensemble methods.

## 📊 Project Overview

This project focuses on classifying deepfake images using various machine learning approaches, starting from traditional SVM models to advanced CNN architectures and ensemble methods. The goal was to achieve high accuracy in distinguishing between different types of synthetic images.

## 🏗️ Model Architectures

### 1. Baseline Model (SVM)
- **Algorithm**: Support Vector Machine with RBF kernel
- **Preprocessing**: Image flattening + StandardScaler normalization
- **Performance**: 63.36% accuracy

### 2. Custom CNN (DeepFakeNet)
- **Architecture**: 3 convolutional blocks with increasing channels (3→32→64→128)
- **Features**: Batch normalization, Dropout, ReLU activation
- **Input Size**: 128×128 RGB images
- **Data Augmentation**: RandomHorizontalFlip, RandomRotation, ColorJitter
- **Optimizer**: SGD
- **Performance**: 90.16% accuracy

### 3. Improved DeepFakeNet
- **Enhancements**:
  - Switched to Adam optimizer (lr=0.0001)
  - Added class weights (especially for class 4: weight=2.7)
  - Implemented learning rate scheduler (ReduceLROnPlateau)
  - Added Gaussian Blur transformation
  - Normalization: mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
- **Performance**: ~92% accuracy

### 4. VGG16 Transfer Learning
- **Base Model**: Pre-trained VGG16
- **Modifications**: Custom classifier head, weighted loss
- **Advanced Augmentation**: CutMix and MixUp
- **Performance**: 92.08% accuracy

### 5. Ensemble Model
- **Combination**: DeepFakeNet2 + VGG16
- **Method**: Weighted ensemble based on model confidence
- **Class Weights**: Class 0 (1.9), Class 4 (1.7)
- **Final Performance**: 
  - Validation: 93.36% accuracy
  - Test: 93.2% accuracy

## 📈 Results

### Ensemble Model Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|-----------|---------|
| 0 | 0.940 | 0.932 | 0.936 | 250 |
| 1 | 0.936 | 0.940 | 0.938 | 250 |
| 2 | 0.955 | 0.928 | 0.941 | 250 |
| 3 | 1.000 | 1.000 | 1.000 | 250 |
| 4 | 0.841 | 0.868 | 0.854 | 250 |

**Overall Accuracy**: 93.36%

## 🛠️ Technical Features

### Data Preprocessing
- Image resizing to 128×128
- Data augmentation techniques
- Normalization to [-1, 1] range
- Class-balanced sampling

### Training Strategies
- Cross-entropy loss with class weights
- Learning rate scheduling
- Early stopping

### Model Improvements
- Batch normalization for training stability
- Dropout layers for overfitting prevention
- Adaptive learning rate management
- Ensemble methods for performance boosting

## 🚀 Key Findings

1. **CNN Superiority**: Convolutional networks significantly outperform traditional methods like SVM for image classification tasks.

2. **Data Augmentation**: Techniques like RandomHorizontalFlip, ColorJitter, and advanced methods like CutMix/MixUp greatly improve model generalization.

3. **Class Imbalance**: Addressing class imbalance through weighted loss functions was crucial for improving performance on minority classes.

4. **Ensemble Benefits**: Combining multiple models through ensemble methods provided the best results, achieving over 93% test accuracy.

5. **Optimizer Choice**: Adam optimizer consistently outperformed SGD for this task.


## 👨‍💻 Author

Daniil Kononov

---

*This project demonstrates the effectiveness of modern deep learning approaches for image classification tasks, particularly in the challenging domain of deepfake detection.*
