# SMS-spam-detection-Project-
Develop a text classification model that separates normal messages from spam messages and evaluate its performance using appropriate classification metrics.

# SMS Spam Detection Using Deep Learning

## 1. Project Overview

This project develops a text classification system that identifies
whether an SMS message is normal (ham) or spam.

The system uses Natural Language Processing (NLP), TensorFlow/Keras,
and a Bidirectional Long Short-Term Memory (BiLSTM) neural network.

A Streamlit web application allows users to enter messages and view
the model's predictions.

## 2. Objectives

- Classify SMS messages as ham or spam.
- Prepare and analyze an SMS text dataset.
- Build a deep learning text classification model.
- Evaluate the model using classification metrics.
- Compare the deep learning model with a traditional ML baseline.
- Develop an interactive web interface for message prediction.

## 3. Technologies Used

- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- TensorFlow/Keras
- Streamlit

## 4. Dataset

Dataset: UCI SMS Spam Collection

Official source:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The original dataset contains labeled SMS messages:
- ham: legitimate messages
- spam: unwanted messages

## 5. Project Structure

sms spam project/
├── data/
│   └── processed/
│       ├── train.csv
│       └── test.csv
├── models/
│   ├── sms_spam_bilstm.keras
│   └── tfidf_logistic_regression.joblib
├── reports/
├── src/
│   ├── inspect_data.py
│   ├── explore_data.py
│   ├── prepare_data.py
│   ├── train_model.py
│   └── compare_baseline.py
├── app.py
└── requirements.txt

## 6. Installation

Install Python and open the project folder in a terminal.

Create a virtual environment:

    python -m venv .venv

Activate it on Windows PowerShell:

    .venv\Scripts\Activate.ps1

Install the dependencies:

    python -m pip install -r requirements.txt

## 7. How to Run the Project

First, download and extract the UCI dataset into the data folder.

Run the data inspection script:

    python src/inspect_data.py

Explore the dataset:

    python src/explore_data.py

Prepare the training and testing datasets:

    python src/prepare_data.py

Train and evaluate the deep learning model:

    python src/train_model.py

Train and evaluate the traditional ML baseline:

    python src/compare_baseline.py

Launch the web application:

    python -m streamlit run app.py

Open the local URL displayed in the terminal.

## 8. Model Evaluation

Evaluate the models using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Actual evaluation results should be reported from the files
generated during the experiments.

## 9. Limitations

- Predictions depend on the dataset used for training.
- Unfamiliar spam patterns may be misclassified.
- The model does not independently verify whether a sender or link
  is trustworthy.
- The model is intended for educational and research purposes.

## 10. Future Scope

- Explore transformer-based text classification.
- Expand the dataset with newer SMS examples.
- Improve multilingual message detection.
- Add a user interface for comparing multiple models.
- Monitor model performance on newly collected data.
