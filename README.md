# Music Genre Classification

This project implements and compares various machine learning and deep learning models for classifying music genres based on audio features. It includes a Jupyter notebook for model training/evaluation/interpretation and a Streamlit web application for real-time genre prediction from uploaded audio files.

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Feature Extraction](#feature-extraction)
  - [Models Implemented](#models-implemented)
  - [Model Interpretability (SHAP)](#model-interpretability-shap)
  - [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Libraries Used](#libraries-used)
- [Future Enhancements](#future-enhancements)

## Description

The goal of this project is to accurately classify music tracks into their respective genres. Audio features are extracted from music files, and these features are used to train both traditional machine learning classifiers and a deep learning model. A web interface allows users to upload their own music files and receive genre predictions from the trained models. Model interpretability using SHAP is explored to understand feature contributions.

## Features

-   **Model Training & Evaluation:** Trains and evaluates multiple classifiers using features from the GTZAN dataset.
    -   Traditional ML: Random Forest, SVM, KNN, Logistic Regression, XGBoost (with hyperparameter tuning using GridSearchCV).
    -   Deep Learning: A Sequential Neural Network using Keras/TensorFlow.
-   **Model Interpretability:** Utilizes SHAP (SHapley Additive exPlanations) to explain model predictions and understand feature importance for both the XGBoost and Neural Network models.
-   **Web Application:** A Streamlit app (`app.py`) for interactive genre prediction.
    -   Accepts user-uploaded WAV or MP3 files.
    -   Performs real-time feature extraction using `librosa`.
    -   Loads pre-trained models (`.pkl` for ML, `.h5` for NN).
    -   Displays the predicted genre (highest confidence) and detailed probabilities from all models.
    -   Shows extracted audio features.

## Dataset

The models were trained using the **GTZAN Genre Collection** dataset features. Specifically, the features pre-extracted and stored in `Data/features_3_sec.csv` (or a similar file path structure as used in the notebook) were utilized. This dataset typically consists of 10 genres with 100 audio tracks each, often segmented into 3-second clips.

## Methodology

### Feature Extraction

The features used for training (present in the CSV) and extracted in real-time by the Streamlit app include:
-   Mel-Frequency Cepstral Coefficients (MFCCs)
-   Chroma Features (chroma_stft)
-   Root Mean Square (RMS) Energy
-   Spectral Centroid, Bandwidth, Rolloff
-   Zero-Crossing Rate
-   Harmonic and Perceptual Features
-   Tempo

Feature extraction is performed using the `librosa` library. Features are scaled using `StandardScaler` before being fed into the models.

### Models Implemented

1.  **Random Forest Classifier**
2.  **Support Vector Machine (SVM)**
3.  **K-Nearest Neighbors (KNN)**
4.  **Logistic Regression**
5.  **XGBoost Classifier**
6.  **Sequential Neural Network (Dense Layers with Dropout)**

Models 1-5 are tuned using `GridSearchCV`. The Neural Network utilizes `EarlyStopping` and `ModelCheckpoint`.

### Model Interpretability (SHAP)

SHAP (SHapley Additive exPlanations) is implemented in the Jupyter notebook (`Music_Genre_Classification.ipynb`) to provide insights into model predictions:
-   **XGBoost:** `shap.TreeExplainer` is used to calculate SHAP values, generating plots (like feature importance bar plots) to show the contribution of each feature to the model's output.
-   **Neural Network:** `shap.DeepExplainer` is used to estimate SHAP values, helping to understand feature influences on the NN's predictions across different classes.

### Evaluation

Models are evaluated based on:
-   Accuracy
-   Classification Report (Precision, Recall, F1-Score)
-   Confusion Matrix
-   (For NN) Log Loss, AUC-ROC (One-vs-Rest)

## Project Structure

![image](https://github.com/user-attachments/assets/80e59885-a172-4ec9-9ae5-6c62fa45052f)


## Setup & Usage

### Prerequisites

-   Python (Version 3.9+ recommended, based on libraries used)
-   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sarojlal/Music_Genre_Classification.git
    cd https://github.com/sarojlal/Music_Genre_Classification.git
    ```
2.  **(Recommended)** Create and activate a virtual environment:
    ```bash
    python -m venv venv
    # On Windows: venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate
    ```
3.  **Install dependencies:**
    *(Create a `requirements.txt` file first if you don't have one)*
    ```bash
    pip install -r requirements.txt
    ```
    *If you don't have a `requirements.txt`, you'll need to manually install the libraries listed below.*

### Running the Application

1.  Ensure the trained models (`.pkl`, `.h5` files) and preprocessing objects (`scaler.pkl`, `label_encoder.pkl`) are available in the correct path expected by `app.py` (e.g., a `saved_models` directory). If not, you may need to run the `Music_Genre_Classification.ipynb` notebook first to generate them.
2.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
3.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
4.  Upload a WAV or MP3 audio file to get the genre prediction.

## Libraries Used

-   `numpy`
-   `pandas`
-   `scikit-learn`
-   `xgboost`
-   `tensorflow` / `keras`
-   `librosa`
-   `streamlit`
-   `matplotlib` / `seaborn` (for plotting in notebook)
-   `pickle` (for saving/loading models)
-   `shap` (for model interpretation)


## Future Enhancements

-   Explore using Mel Spectrograms directly as input to CNN/Transformer models.
-   Implement real-time audio recording and classification within the Streamlit app.
-   Enhance UI/UX, add audio playback, and improve visualization of results.
-   Containerize the application using Docker for easier deployment.
-   Collect user feedback on predictions for model improvement.
