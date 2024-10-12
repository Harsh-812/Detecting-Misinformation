# Detecting-Misinformation

## Project Overview

The goal of this project is to build a machine learning model to detect misinformation in news articles. It uses different machine learning algorithms, including Logistic Regression, Naive Bayes, and Support Vector Machines (SVM), as well as deep learning techniques with LSTMs.

## Key Features

- **Natural Language Processing (NLP)**: The project leverages several NLP techniques, including tokenization, stopword removal, stemming, and lemmatization, to preprocess the text data and prepare it for model training.
- **Word Clouds**: Visualizes the most frequent words in the dataset to understand common patterns in true and fake news articles, helping to identify key features for classification.
- **Model Training**: Various machine learning and deep learning models are trained to classify news articles, including traditional machine learning algorithms like Naive Bayes and Logistic Regression, alongside deep learning models such as LSTMs.
- **Performance Evaluation**: Model performance is rigorously evaluated using accuracy, precision, recall, F1 score, and confusion matrices, providing insights into the effectiveness of each model.
- **Model Saving**: Trained models are saved using Joblib, enabling easy deployment for real-time or batch processing scenarios.

## Data

The project utilizes a dataset from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
 containing two CSV files:
- **True.csv**: Contains news articles labeled as true.
- **Fake.csv**: Contains news articles labeled as fake.

## Model Architecture

The deep learning model is based on an LSTM (Long Short-Term Memory) network, which is particularly effective for sequential data like text. /n
The architecture includes:

* Embedding layer to convert words into dense vectors
* LSTM layer for capturing temporal dependencies in the text
* Dense layers for classification

## Results

The project evaluated three models: Logistic Regression, a Basic LSTM Model, and a Tuned LSTM Model. Here are the accuracy results:

- **Logistic Regression:** Achieved the highest accuracy of 99.53%.
- **Basic LSTM Model:** Slightly lower accuracy at 98.48%.
- **Tuned LSTM Model:** Accuracy of 98.46%, comparable to the Basic LSTM.

## Key Observations:

* Logistic Regression outperformed both LSTM models, showcasing that simpler traditional models can sometimes be more effective than deep learning approaches.
* Both the Basic and Tuned LSTM models had similar performance.
* Logistic Regression's high accuracy and simplicity make it an ideal choice for real-time misinformation detection.
