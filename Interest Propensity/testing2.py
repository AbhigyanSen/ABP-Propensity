from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('/home/abp/Documents/Interest/RFModel.pkl')

# Define categorical columns
categorical_cols = ['Gender_Source', 'City_Source', 'Occupation_Source',  'Income_Source', 'HighestEducation_Source',
                    'ProfileCreatedBy_Source', 'MaritalStatus_Source', 'PhysicalAppearance_Source',
                    'Complexion_Source', 'LivingHouseType_Source', 'familyType_Source', 'Caste_Source', 'Mangalik_Source',
                    'Gender_Target', 'City_Target', 'Occupation_Target', 'Income_Target', 'HighestEducation_Target',
                    'ProfileCreatedBy_Target', 'MaritalStatus_Target', 'PhysicalAppearance_Target',
                    'Complexion_Target', 'LivingHouseType_Target', 'familyType_Target', 'Caste_Target', 'Mangalik_Target']

# Define numerical columns
numerical_cols = ['Age_Source', 'Height_Source', 'NoOfPhotosAdded_Source',
                  'Age_Target', 'Height_Target', 'NoOfPhotosAdded_Target']

# Column transformer for preprocessing
preprocessor = None
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Define the API route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()

    # Convert data into a pandas DataFrame (assuming data is sent as a dictionary)
    input_data = pd.DataFrame(data)

    # Ensure that input data contains the same columns as the training data
    input_data = input_data[categorical_cols + numerical_cols]
    print("A")
    # Preprocess the data using the preprocessor (same preprocessing as in training)
    # processed_data = preprocessor.transform(input_data)  # Use transform, not fit_transform
    print("B")

    # Predict using the trained model
    prediction = model.predict(input_data)
    print("C")
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction.tolist()})


# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)