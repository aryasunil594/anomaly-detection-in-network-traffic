import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load your network traffic dataset (example: CSV format)
data = pd.read_csv('network_traffic.csv')

# Feature selection (Assuming the dataset has columns: 'src_ip', 'dst_ip', 'packet_size', etc.)
X = data[['packet_size', 'duration', 'src_bytes', 'dst_bytes']]  # Example features
y = data['label']  # 1 for anomaly, 0 for normal traffic

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Anomaly detection model (using Isolation Forest)
model = IsolationForest(contamination=0.1)
model.fit(X_train)

# Save the trained model
joblib.dump(model, 'anomaly_detection_model.pkl')

# Test the model
y_pred = model.predict(X_test)
y_pred = np.where(y_pred == 1, 0, 1)  # Convert -1 (outlier) to 1 for anomaly, and 1 to 0 for normal traffic

# Print classification report
print(classification_report(y_test, y_pred))

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained anomaly detection model
model = joblib.load('anomaly_detection_model.pkl')

# Title of the web app
st.title('Anomaly Detection in Network Traffic')

# Input form for user
packet_size = st.number_input('Packet Size', min_value=0, max_value=1500, step=1)
duration = st.number_input('Duration', min_value=0, step=1)
src_bytes = st.number_input('Source Bytes', min_value=0, step=1)
dst_bytes = st.number_input('Destination Bytes', min_value=0, step=1)

# Collect input data for prediction
input_data = np.array([[packet_size, duration, src_bytes, dst_bytes]])

# Button to trigger the prediction
if st.button('Predict Anomaly'):
    prediction = model.predict(input_data)
    if prediction == 1:
        st.write("Prediction: Normal Traffic")
    else:
        st.write("Prediction: Anomaly Detected")
        
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('anomaly_detection_model.pkl')

# Set the title of the app
st.title("Anomaly Detection Web App")

# File uploader to upload data
uploaded_file = st.file_uploader("Upload your data", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", data)

    # Assume the necessary features are selected
    features = data[['sourceip', 'destinationip', 'sourceport', 'destinationport',
                     'protocol', 'bytessent', 'bytesreceived', 'packetssent',
                     'packetsreceived', 'duration']]

    # Get predictions from the model
    predictions = model.predict(features)
    data['Prediction'] = predictions

    # Display the predictions
    st.write("Predictions (1 = Anomaly, -1 = Normal):")
    st.dataframe(data)
