AI/ML Projects - Shubham Mallick
Overview
This repository contains various AI and Machine Learning projects that I have worked on. The goal of these projects is to develop an understanding of various ML algorithms and their real-world applications.

Contents
Dataset: The dataset used for training and testing the models.
Preprocessing: Steps for data cleaning, transformation, and feature selection.
Models: Implementation of different machine learning models (e.g., Random Forest, Logistic Regression).
Evaluation: Metrics and methods used to evaluate the models' performance.
Future Work: Plans for future improvements.
Dataset
The dataset used in this project is the Algerian Forest Fires dataset, which contains various meteorological and environmental features, such as temperature, humidity, wind speed, rain, and various fire intensity indices. The target variable is Classes, which indicates whether a fire occurred (fire) or not (not fire).

Dataset Columns:
Temperature: Temperature during the observation period.
RH: Relative Humidity.
Ws: Wind speed.
Rain: Rainfall during the observation period.
FFMC, DMC, DC, ISI, BUI, FWI: Various fire weather indices.
Classes: The target variable (fire or not fire).
Region: The region in which the observation was made.
Preprocessing
Handling Missing Data: All missing values were filled with the mean of the respective columns.
Feature Selection: We dropped non-essential columns such as day, month, year, and Region.
Data Conversion: The target variable Classes was mapped to integers (0 for not fire, 1 for fire).
Models Used

1. Random Forest Classifier:
   A robust ensemble method used for classification tasks. It is known for handling large datasets with higher accuracy and preventing overfitting.

2. Logistic Regression:
   A classical model used for binary classification tasks. It models the relationship between the dependent and independent variables using a logistic function.

3. Train-Test Split:
   The dataset is split into training and testing sets using the train_test_split function, with 80% of the data used for training and 20% for testing.

4. Scaling:
   The features are scaled using MinMaxScaler to bring all variables to the same scale before training.

5. Evaluation Metrics:
   Accuracy Score
   Confusion Matrix
   Classification Report (Precision, Recall, F1-score)
   ROC Curve and AUC Score
   Model Evaluation
   After training the models, various evaluation metrics were calculated:

Accuracy: The overall accuracy of the model.
Precision and Recall: These metrics help measure the performance of the model, especially in imbalanced datasets.
Confusion Matrix: This helps us understand the true positive, false positive, true negative, and false negative predictions.
Future Work
Hyperparameter Tuning: Tuning the hyperparameters of the models to improve their performance.
Ensemble Methods: Implementing ensemble techniques such as stacking or boosting.
Additional Features: Incorporating additional features to improve model accuracy.
