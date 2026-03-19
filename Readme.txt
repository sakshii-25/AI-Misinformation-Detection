Evaluating the Usefulness of AI/ML in Misinformation Detection and Mitigation in Social Media  

Overview
This project focuses on building a machine learning model to classify real and fake news using textual data from tweets. It employs a combination of TF-IDF, Latent Semantic Analysis (LSA), and Word2Vec embeddings to extract features and leverages various machine learning algorithms such as Logistic Regression, K-Nearest Neighbors (KNN), Random Forest, Decision Tree, and XGBoost for classification. Additionally, a hybrid approach integrating the Google Fact-Check API enhances the model's prediction accuracy by cross-verifying news authenticity.

Prerequisites
Before running the code, ensure you have the following installed:

Python 3.x
Required libraries:
numpy
pandas
scikit-learn
gensim
xgboost
matplotlib
seaborn
requests
nltk
wordcloud
bs4 (BeautifulSoup for HTML parsing)
contractions
If you are running the code on Google Colab, make sure to install the required libraries using:

Upload the dataset (data.csv) to Colab using:

python
Copy code
from google.colab import files
uploaded = files.upload()
Load the data into a Pandas DataFrame:

python
Copy code
import pandas as pd
df = pd.read_csv('data.csv')
Now, you can proceed with running the rest of the code for feature extraction, model training, and evaluation.

Running Locally
If you are running the code on your local machine:

Clone the repository or download the project files.
Place the dataset (data.csv) in the project directory.
Load the dataset as follows:
python
Copy code
import pandas as pd
df = pd.read_csv('data.csv')

Project Structure
Feature Extraction: This step includes TF-IDF, LSA (Truncated SVD), and Word2Vec for textual data representation.
Machine Learning Models: The project implements several models for fake news classification, including:
Logistic Regression
K-Nearest Neighbors (KNN)
Random Forest
Decision Tree
XGBoost

Hybrid Approach: The system also integrates the Google Fact-Check API to enhance accuracy by cross-verifying the model predictions.

How to Run
Load the dataset (data.csv).

Preprocess the data using the provided code for cleaning, tokenizing, and vectorizing.
Train the models using the provided algorithms (e.g., Logistic Regression, Random Forest, XGBoost).
Use the Google Fact-Check API to cross-verify predictions.

Evaluate the model using metrics like accuracy, precision, recall, and F1-score.
Evaluation Metrics
The project evaluates models using common metrics, including:

Accuracy
Precision
Recall
F1-Score
Bar graphs and ROC curves are generated to compare the performance of the different models.

Notes
Google Fact-Check API: I have added a valid API key from Google to use the fact-checking features.
Feature Extraction: Ensure that the same preprocessing and feature extraction steps are applied consistently throughout the model training and evaluation phases.

