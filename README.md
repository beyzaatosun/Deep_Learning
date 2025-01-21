# Deep Learning Projects

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
   
- In this project, a Convolutional Neural Network (CNN) model is developed to classify images from the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 different classes.
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

3. **Sentiment Analysis on the IMDB Dataset Using an RNN Model**

- This project uses TensorFlow and Keras Tuner to create an RNN (Recurrent Neural Network) model for sentiment analysis on the IMDB dataset. In the project, Keras Tuner is used to optimize the model’s hyperparameters using the Random Search method.

  - **Data Steps:**
    - Loading the IMDB Dataset: The IMDB dataset is loaded using Keras's imdb.load_data() function. This dataset contains labeled movie reviews to perform sentiment analysis. The reviews are converted into numbers using a vocabulary of the top 10,000 words.

    - Data Preprocessing: The loaded text data is padded using the pad_sequences function, with a maximum word length of 100. This ensures that the input data has a consistent length for model processing.

  - **Model Architecture Definition:**

     - **Embedding Layer:** Represents words as low-dimensional dense vectors (embeddings).
     - **SimpleRNN Layer:** The RNN processes sequential data and provides an output for each input.
     - **Dropout Layer:** Helps prevent overfitting by increasing the model’s generalization ability.
     - **Dense Layer:** The final layer of the model, using a sigmoid activation function for sentiment prediction.
     ```python
    def build_model(hp):
    model = Sequential()
    #embedding kelimeleri vektörlere cevirir
    model.add(Embedding(input_dim=max_features,
                        output_dim=hp.Int("embedding_output", min_value=32, max_value=128, step=32),
                        input_length=max_len))
    model.add(SimpleRNN(units=hp.Int("rnn_units", min_value=32, max_value=128,step=32)))
    model.add(Dropout(rate=hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(optimizer=hp.Choice("optimizer", ["adam","rmsprop"]),
                                     loss="binary_crossentropy",
                                     metrics=["accuracy"])
    return model
    
  - **Hyperparameter Search:**
    - Keras Tuner is used to perform hyperparameter optimization with RandomSearch.
    - Hyperparameters such as embedding output size, RNN units, dropout rate, and optimizer are optimized.
    - The RandomSearch method is used to find the best hyperparameter combination.
      ```python
      tuner = RandomSearch(
      build_model, 
      objective = "val_loss" , 
      max_trials=2, 
      executions_per_trial=2, 
      directory="rnn_tuner_directory",
      project_name="rnn"
      )
      
  - **Early Stopping:**

    Early stopping is applied, which terminates training when the validation loss does not improve. The best model weights are restored after training.
    ```python
    early_stopping = EarlyStopping(monitor ="val_loss", patience=3, restore_best_weights=True)
    tuner.search(x_train, y_train,
             epochs=15,
             validation_split=0.2,
             callbacks=[early_stopping])
    

