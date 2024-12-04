from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from sklearn.metrics import classification_report

from ddlists import Gender as ge, City as ci, Occupation as  occ, Income as inc, Education as edu, profilecreatedBy as pcb, maritalStatus as ms

labelmap = {
    "Gender" : ge,
    "City" : ci,
    "Occupation" :  occ,
    "Income" : inc,
    "Education" : edu,
    "profilecreatedBy" : pcb ,
    "maritalStatus" : ms
}

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess the data
def load_and_train_model():
    # Load the CSV file into a DataFrame
    cdf = pd.read_csv("/home/abp/Documents/ModelSales/Sales.csv")
    cdf = cdf.iloc[:20000]
    # Drop specified columns
    # nnfeature = ["UserID","awid",'FirstPlanPurchase','ProfileCreationDate']
    nnfeature = ["UserID","awid",'FirstPlanPurchase','ProfileCreationDate', 'InterestSent', 'SalesTeamCallMade', 'City']
    cdf = cdf.drop(columns=nnfeature)

    # Replace null values in 'PlanPurchase' with 0, and others with 1
    cdf['PlanPurchase'] = cdf['PlanPurchase'].fillna(0).apply(lambda x: 1 if x > 0 else 0)
    for column in ['Education','Occupation','Income']:
        cdf[column] = cdf[column].fillna('None')
    cdf = cdf.dropna()

    # Step 5: Replace null values in 'Height' based on 'Gender'
    cdf.loc[(cdf['Gender'] == 'Female') & (cdf['Height'].isnull()), 'Height'] = '4.80 feet'
    cdf.loc[(cdf['Gender'] == 'Male') & (cdf['Height'].isnull()), 'Height'] = '4.91 feet'

    # Optionally convert Height back to float format if needed
    cdf['Height'] = cdf['Height'].str.replace(' feet', '').astype(float)

    # Define features and target variable
    X = cdf.drop(columns=['PlanPurchase'])
    y = cdf['PlanPurchase']

    # Convert categorical columns to numerical using Label Encoding
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        le = le.fit(labelmap[column])
        X[column] = le.transform(X[column])
        label_encoders[column] = le

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the scaled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train a Logistic Regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    # Return the trained model, scaler, and label encoders
    return model, scaler, label_encoders

# Load the model and preprocess data
model, scaler, label_encoders = load_and_train_model()

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    input_data = request.json

    # Check if any input fields are missing
    required_fields = ['NoofLogin', 'ProfileViews', 
                       'Gender', 'Age', 'Income', 'Education', 
                       'Occupation', 'profilecreatedBy', 'maritalStatus', 'Height']
    
    for field in required_fields:
        if field not in input_data:
            return jsonify({'error': 'Please fill all the input fields'}), 400

    # Create DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    try:    
        # Explicit conditions for PlanPurchase
        if input_df['Age'].iloc[0] < 18 or input_df['Age'].iloc[0] > 100:
            predicted_plan_purchase = 0 
        else:
            # Transform categorical columns
            for column in input_df.select_dtypes(include=['object']).columns:
                if column in label_encoders:
                    input_df[column] = label_encoders[column].transform(input_df[column])

            # Scale the features
            input_scaled = scaler.transform(input_df)

        # Predict
        predicted_plan_purchase = model.predict(input_scaled)[0]
        predicted_plan_purchase_proba = model.predict_proba(input_scaled)[0]
        prob = max(predicted_plan_purchase_proba)
        prob = round(prob, 2)
        

        # print(f"1: {predicted_plan_purchase}")
        # print(f"2: {max(predicted_plan_purchase_proba)}")

        # Convert prediction to boolean
        result = bool(predicted_plan_purchase)
        # print(f"Res: {result}")

        if result == True:
            prob = prob
        elif result == False:
            prob = -1 * prob
        else:
            prob = "error"
        
        return jsonify({'PlanPurchase': prob})
    
    except Exception as e:
        return jsonify({'PlanPurchase': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)