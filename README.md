# **Tax Fraud Detection System**
![alt text](https://github.com/Ahmedsamy96/Taxpayer-Fraud-Detection/blob/main/fraud.png)

<p>This repository contains a system for detecting potential tax fraud in financial data.</p>

## Project Structure
### The project consists of two main files:

- **data_pipeline.ipynb:** Jupyter notebook containing the data generation, exploration, feature engineering, model training, and evaluation pipeline.
- **app.py:** Streamlit application for deploying the trained model as a web app for real-time fraud prediction.
**Note:** The trained_model.pkl file generated by the Jupyter notebook is not included in the repository due to its potential size.

## Data Pipeline (data_pipeline.ipynb)
### This Jupyter notebook performs the following tasks:

- **Data Generation:**
Simulates a dataset of financial transactions with features like income, expenses, tax liability, and fraud indicators.
- **Data Exploration:**
Analyzes the generated data to understand the relationships between features and potential fraud.
- **Feature Engineering:**
Creates a "Fraud" feature using anomaly detection techniques.
- **Model Training:**
Trains a Random Forest classification model to predict tax fraud based on financial data.
- **Model Evaluation:**
Evaluates the performance of the trained model using metrics like accuracy, precision, recall, F1 score, and ROC-AUC score.
- **Model Saving:**
Saves the trained model as trained_model.pkl for later use in the web app.

## Web App (app.py)
### This Streamlit application allows users to:

1. Upload a CSV file containing financial transaction data.
2. View the uploaded data.
3. Make real-time predictions on whether each transaction is likely fraudulent using the trained model.
4. Download the predicted data with a new "Predicted Fraud" column.
5. Explore basic Exploratory Data Analysis (EDA) visualizations of the uploaded data, including:
6. Value counts of predicted fraud
7. Correlation heatmap
**Note:** This web app requires the trained_model.pkl file to be present in the same directory for loading the trained model.

## Getting Started
### Prerequisites:

Python 3.x
Jupyter Notebook
Streamlit
pandas
matplotlib
seaborn
scikit-learn
pickle

### Instructions:
- Clone this repository.
- Install the required libraries using pip install -r requirements.txt (assuming you have a requirements.txt file listing the dependencies).
- For data pipeline:
  - Open data_pipeline.ipynb in Jupyter Notebook and run all the cells to generate data, train the model, and save it.
- For web app:
  - Run streamlit run app.py from the command line in the project directory.
  - This will launch the Streamlit app in your web browser, allowing you to upload a CSV file and view predictions.

## Additional Notes:
- You can replace the simulated data generation in the Jupyter notebook with your actual financial data for training the model.
- The web app provides a basic set of EDA visualizations. You can customize it further to include additional visualizations based on your needs.
