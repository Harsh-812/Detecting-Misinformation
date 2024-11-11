# Detecting-Misinformation

## Project Overview

The goal of this project is to build a machine learning model to detect misinformation in news articles. It uses different machine learning algorithms, including Logistic Regression, Naive Bayes, and Support Vector Machines (SVM).

## Key Features

- **Natural Language Processing (NLP)**: The project leverages several NLP techniques, including tokenization, stopword removal, stemming, and lemmatization, to preprocess the text data and prepare it for model training.
- **Word Clouds**: Visualizes the most frequent words in the dataset to understand common patterns in true and fake news articles, helping to identify key features for classification.
- **Model Training**: Various machine learning models are trained to classify news articles, including classification algorithms like Naive Bayes, Support Vector machine and Logistic Regression.
- **Performance Evaluation**: Model performance is rigorously evaluated using accuracy, precision, recall and F1 score, providing insights into the effectiveness of each model.

## Data

The project utilizes a dataset from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
 containing two CSV files:
- **True.csv**: Contains news articles labeled as true.
- **Fake.csv**: Contains news articles labeled as fake.


## Results

The project evaluated three models: Logistic Regression, Naive Bayes and SVM. Here are the accuracy results:

<img width="461" alt="image" src="https://github.com/user-attachments/assets/afe3869e-17ff-4c45-9b0a-862584debea3">


## Key Observations:

* Logistic Regression and SVM Achieve Nearly Perfect Performance: Both Logistic Regression and SVM demonstrate extremely high accuracy (1.00), precision, recall, and F1-scores across both classes. This indicates they are highly effective at distinguishing between true and false information in this dataset.

* Naive Bayes Performs Slightly Lower: Naive Bayes has an accuracy of 0.95, with lower precision and recall, especially for the positive class (1.0). This suggests that Naive Bayes may not be as effective at capturing the nuances in language patterns that differentiate true from false information, possibly due to its simpler, probabilistic approach.

* Balanced Performance Across Classes: All models show similar performance for both classes (0 and 1), with balanced precision, recall, and F1-scores. This indicates that the models are not biased towards one class and are equally effective at identifying true and false information, which is important for a balanced misinformation detection system.
