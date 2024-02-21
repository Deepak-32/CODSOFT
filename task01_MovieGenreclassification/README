Main Objective: Movie Genre Classification,Prediction
This folder contains Python code for a movie genre prediction system using MultiOutput Naive Bayes. It predicts movie genres based on their plots.
The dataset is loded from : https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb

Libraries and Techniques Used
Pandas (pandas) for data manipulation and analysis.
NumPy (numpy) for numerical operations.
Scikit-Learn (scikit-learn) for machine learning tools, including TfidfVectorizer, MultiOutputClassifier, MultinomialNB, and MultiLabelBinarizer.
TQDM (tqdm) for displaying progress bars during data processing.
Code Overview
Import necessary libraries: Import all the required libraries for the project, including pandas, numpy, scikit-learn components, and tqdm for progress bars.
Define the list of genres: Create a list of movie genres that will be used for classification.
Define a fallback genre: Specify a fallback genre to assign to movies with no predicted genre.
Load training data: Load the training data from train_data.txt using pandas, where each row contains a movie serial number, name, genre(s), and plot.
Data preprocessing for training data: Prepare the training data by converting movie plot text to lowercase and encoding genre labels using MultiLabelBinarizer.
TF-IDF Vectorization: Use TfidfVectorizer to convert the movie plot text into TF-IDF features.
Train the MultiOutput Naive Bayes classifier: Train a MultiOutputClassifier using a Multinomial Naive Bayes classifier with the training data.
Load test data: Load the test data from test_data.txt using pandas, which includes movie serial number, name, and plot.
Data preprocessing for test data: Preprocess the test data by converting movie plot text to lowercase.
Vectorize the test data: Transform the test data using the same TF-IDF vectorizer used for training.
Predict genres on test data: Predict movie genres on the test data using the trained model.
Create a results DataFrame: Create a DataFrame that contains movie names and predicted genres.
Replace empty predicted genres: Replace empty predicted genres with the fallback genre.
Write results to a text file: Save the prediction results to model_evaluation.txt with proper formatting and UTF-8 encoding.
Calculate evaluation metrics: Calculate evaluation metrics, including accuracy, precision, recall, and F1-score, using training labels as a proxy.
Append metrics to the output file: Append the evaluation metrics to the model_evaluation.txt file.
Usage
Specify the configuration file (config_file) and frozen model file (frozen_model) for the pre-trained model.
Load class labels from a text file (file_name) to recognize objects.
Configure model input size, scaling, and color format.
Load an image or a video for object detection and recognition.
Display the recognized objects, their bounding boxes, and confidence scores.
For video, real-time object detection and recognition are performed.
TL;DR
This code utilizes MultiOutput Naive Bayes to predict movie genres based on plot descriptions, involving data preprocessing, TF-IDF vectorization, and evaluation metrics calculation. To use the code, ensure required libraries are installed, provide training and test data, run the code, and review genre predictions and evaluation metrics in model_evaluation.txt.

Requirements
Python 3.x
Pandas (pandas)
NumPy (numpy)
Scikit-Learn (scikit-learn)
TQDM (tqdm)
