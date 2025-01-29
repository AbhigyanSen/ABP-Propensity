from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load pre-trained models and encoders
model = joblib.load('decision_tree_model.pkl')
onehot_encoders = joblib.load('onehot_encoders.pkl')
scaler = joblib.load('standard_scaler.pkl')

# Load city cleaned data to map city to type
city_df = pd.read_csv('city_data.csv')
city_df['Place'] = city_df['Place'].str.strip().str.lower()         # Normalize city names

from ddlist import Gender as ge, Income as inc, \
      Education as edu, profilecreatedBy as pcb, maritalStatus as ms, \
      CityType as ct

labelmap = {
    "Gender": ge,
    "Education": edu,
    "ProfileCreatedBy": pcb,
    "maritalStatus": list(set(ms)),
    "CityType": ct
}

app = Flask(__name__)
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": 100})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        city = input_data.get("City", "").strip().lower()       # Normalising city name

        # City Type
        city_type = city_df.loc[city_df['Place'] == city, 'Type'].iloc[0] if city in city_df['Place'].values else "Unknown"
        input_data["CityType"] = city_type


        df = pd.DataFrame([input_data])
        df["Income"] = df["Income"].map(inc)
        df = df.drop(["City"],axis=1)
        
        transformed_data = pd.DataFrame()
        for column in df.select_dtypes(include=['object']).columns:
            if column not in onehot_encoders:
                continue
            
            ohe = onehot_encoders[column]
            transformed = ohe.transform(df[[column]])
            onehot_columns = [f"{column}_{category}" for category in labelmap[column]]
            transformed_df = pd.DataFrame.sparse.from_spmatrix(transformed, columns=onehot_columns, index=df.index)
            transformed_data = pd.concat([transformed_data, transformed_df], axis=1)

        numerical_data = df.select_dtypes(exclude=['object'])
        newXEncoded = pd.concat([numerical_data, transformed_data], axis=1)
        X_scaled = scaler.transform(newXEncoded)

        y_pred = model.predict(X_scaled)
        return jsonify({"prediction": int(y_pred[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)