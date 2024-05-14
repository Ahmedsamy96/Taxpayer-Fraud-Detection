import pickle
import streamlit as st
import pandas as pd
import requests
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
#model_path = "https://github.com/Ahmedsamy96/Taxpayer-Fraud-Detection/blob/main/trained_model.pkl"
# Load the trained model from GitHub
model_url = "https://github.com/Ahmedsamy96/Taxpayer-Fraud-Detection/raw/main/trained_model.pkl"
model_response = requests.get(model_url)
model_response.raise_for_status()  # Check if download was successful
model = pickle.loads(model_response.content)


# Define a function to preprocess the input data
def preprocess_data(df):
    # Assuming df is the input DataFrame
    # Dropping any existing 'Tax Evasion - Fraud' column
    if 'Tax Evasion - Fraud' in df.columns:
        df.drop('Tax Evasion - Fraud', axis=1, inplace=True)
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    return X_scaled

# Define the Streamlit app
def main():
    st.title("Tax Fraud Detection App")
    st.write("Upload a CSV file with transaction data to predict fraud.")

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Original Data")
        st.write(data.head())

        # Preprocess the data
        processed_data = preprocess_data(data)

        # Make predictions
        predictions = model.predict(processed_data)

        # Add predictions to the original DataFrame
        data['Predicted Fraud'] = predictions
        st.write("### Predicted Data")
        st.write(data.head(15))

        # Download the predicted data as CSV
        st.download_button(
            label="Download Predicted Data as CSV",
            data=data.to_csv().encode(),
            file_name="predicted_data.csv",
            mime="text/csv"
        )
        st.markdown("<hr>", unsafe_allow_html=True)


        # Perform EDA
        st.write("### EDA Results:")
        st.write("- Value Counts for Fraud Predictions:")
        fraud_counts = data['Predicted Fraud'].value_counts()
        st.bar_chart(fraud_counts)
        st.markdown("<hr>", unsafe_allow_html=True)


        # Display other important EDA results
        st.write("- Correlation Heatmap:")
        st.write(data.corr())
        sns.heatmap(data.corr(), annot=True)
        plt.show()
        st.markdown("<hr>", unsafe_allow_html=True)


        # Create a report
        report = generate_report(data, fraud_counts)
        st.markdown(report, unsafe_allow_html=True)

# Function to generate a report
def generate_report(data, fraud_counts):
    report = """
    <h2>Report</h2>
    <h3>Summary</h3>
    <p>Total Transactions: {}</p>
    <p>Total Predicted Fraudulent Transactions: {}</p>
    <p>Total Non-Fraudulent Transactions: {}</p>
    """.format(len(data), fraud_counts.get(1, 0), fraud_counts.get(0, 0))
    
    return report    

if __name__ == "__main__":
    main()
