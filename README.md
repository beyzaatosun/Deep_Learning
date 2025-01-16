# Deep_Learning Projects

This repository is a collection of various projects that utilize deep learning techniques. Each project uses a different model and dataset to apply deep learning concepts. This repo serves as a starting point for those who want to practice and learn about deep learning methods.

## Projects:
1.**ANN - MNIST Classification:**
- In this project, an Artificial Neural Network (ANN) model is developed to classify handwritten digits from the MNIST dataset.
- **Technologies Used:** Keras, TensorFlow, NumPy, Matplotlib
- **Model Architecture:**
  - Input: 28x28 pixel grayscale images
  - Layers: 2 fully connected (dense) layers and 1 output layer
  - Activation Functions: ReLU and Softmax
  - Loss Function: Categorical Crossentropy
  - Optimizer: Adam
- **Model Performance:**
  - Test Accuracy: %97
  - Test Loss: 0.083

2. **CNN - CIFAR-10 Image Classification:**
In this project, a Convolutional Neural Network (CNN) model is developed to classify images from the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 different classes.
- **Technologies Used:** Keras, TensorFlow, NumPy, Matplotlib, Scikit-learn
- **Model Architecture:**
  - Input: 32x32 color images (RGB)
  - Layers: Convolutional layers for feature extraction, Pooling layers for dimensionality reduction,, Dense layers for classification,Output layer with softmax activation to predict 10 classes
  - Optimizer: RMSprop with a learning rate of 0.0001 and decay of 1e-6.
  - Loss Function: Categorical Crossentropy.
- **Model Performance:**
  - Test Accuracy: 75.25%
  - Test Loss: 0.7563
- **Data Preprocessing:**
  - Normalization: The pixel values of the images are scaled to the range [0, 1] by dividing by 255.
  - One-hot Encoding: The labels are encoded into one-hot vectors.
  - Image Augmentation: Rotation range, width/height shift, shear range, zoom range, and horizontal flip are used to augment the training data for better generalization.


