
# Facial Emotion Recognition (FER) using CNN

## 📌 Introduction
Facial expressions are one of the most powerful and universal forms of non-verbal communication.  
They can convey emotions such as happiness, sadness, anger, fear, surprise, and more without the need for words.

The goal of this project is to build a **Facial Emotion Recognition (FER)** system that can automatically detect a person’s emotional state from an image using **Deep Learning**.  

We use the **FER-2013 dataset**, which contains grayscale facial images categorized into **seven emotions**:
- Angry 😠
- Disgusted 🤢
- Fearful 😨
- Happy 😃
- Neutral 😐
- Sad 😢
- Surprised 😲

---

## 🎯 Objectives
- Preprocess and scale the dataset for model training.
- Build and train a **Convolutional Neural Network (CNN)** to classify facial emotions.
- Evaluate the model’s performance on unseen data.
- Suggest potential improvements for higher accuracy.
- Prepare the model for possible real-time emotion recognition applications.

---

## 📂 Dataset
The dataset used is **[Emotion Detection FER](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)**, a cleaned and preprocessed version of FER-2013.  
It contains:
- **Train set:** 28,709 images
- **Test set:** 7,178 images

---

## 🛠️ Steps Followed
1. **Data Loading** – Downloaded from Kaggle using `kaggle.json`.
2. **Data Preprocessing** – Rescaled pixel values to `[0, 1]`, converted to grayscale, and split into training/validation sets.
3. **Data Augmentation** – Applied transformations like rotation, zoom, and horizontal flips to reduce overfitting.
4. **Model Building** – Created a CNN architecture with multiple convolutional and pooling layers, followed by dense layers.
5. **Training** – Trained with `categorical_crossentropy` loss and `Adam` optimizer, using early stopping and model checkpoint.
6. **Evaluation** – Measured accuracy on the test set and visualized predictions.
7. **Visualization** – Plotted training/validation accuracy and loss graphs.

---

## 📊 Model Performance
- **Test Accuracy:** ~61.02%
- **Loss & Accuracy Graphs:** Included in notebook for visualization.
- **Challenges:** Model accuracy can be improved with deeper architectures, transfer learning, and hyperparameter tuning.

---

## 🚀 Future Improvements
- Use **Transfer Learning** (e.g., VGG16, ResNet50, EfficientNet).
- Increase training epochs with a learning rate scheduler.
- Improve data augmentation strategies.
- Collect higher-resolution images for training.

---

## 📦 Requirements
- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- Kaggle API
- Google Colab or Jupyter Notebook
