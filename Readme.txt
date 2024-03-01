Import necessary libraries
import pandas as pd # For data manipulation and analysis from sklearn.model_selection import train_test_split # For splitting data into training and testing sets from sklearn.linear_model import LogisticRegression # For Logistic Regression model from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score) # For model evaluation metrics Import Libraries: In this section, we import the required libraries for working with data, building machine learning models, and evaluating their performance.

Load the dataset
load it from link : https://www.kaggle.com/datasets/kartik2112/fraud-detection data = pd.read_csv('fraudTrain.csv') data = pd.read_csv('fraudTest.csv') Load Dataset: We load the dataset containing information about credit card transactions.

Prepare the data (handle missing values, encode categorical variables, scale features)
This step is typically done before splitting the data into training and testing sets.
It includes handling missing values, encoding categorical variables, and scaling features if necessary.
This part is omitted in the provided code for brevity.
Split data into features (X) and target variable (y)
X = data.drop('fraudulent', axis=1) # Features (independent variables) y = data['fraudulent'] # Target variable (dependent variable) Split Data: We split the data into features (X) and the target variable (y). The features are the attributes of credit card transactions (e.g., amount, merchant), and the target variable is whether the transaction is fraudulent or not.

Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) Train-Test Split: We split the data into training and testing sets. The training set (80%) is used to train the model, and the testing set (20%) is used to evaluate the model's performance.

Initialize and train the Logistic Regression model
model = LogisticRegression() # Initialize Logistic Regression model model.fit(X_train, y_train) # Train the model on the training data Model Training: We initialize a Logistic Regression model and train it using the training data (X_train and y_train).

Make predictions on the testing set
y_pred = model.predict(X_test) Prediction: We make predictions on the testing set (X_test) using the trained model.

Evaluate the model
accuracy = accuracy_score(y_test, y_pred) precision = precision_score(y_test, y_pred) recall = recall_score(y_test, y_pred) f1 = f1_score(y_test, y_pred) roc_auc = roc_auc_score(y_test, y_pred) Model Evaluation: We evaluate the performance of the model using various evaluation metrics such as accuracy, precision, recall, F1-score, and ROC AUC score.

Print model evaluation metrics
print("Model Evaluation:") print("Accuracy:", accuracy) print("Precision:", precision) print("Recall:", recall) print("F1 Score:", f1) print("ROC AUC Score:", roc_auc) Print Results: Finally, we print the evaluation metrics to assess the model's performance.