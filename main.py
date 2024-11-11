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


# Load and preprocess the data
def load_and_train_model():
    # Load the CSV file into a DataFrame
    cdf = pd.read_csv("/home/abp/Documents/ModelSales/Sales.csv")
    cdf = cdf.iloc[:20000]
    # Drop specified columns
    nnfeature = ["UserID", "awid", 'FirstPlanPurchase', 'ProfileCreationDate']
    cdf = cdf.drop(columns=nnfeature)

    # Replace null values in 'PlanPurchase' with 0, and others with 1
    cdf['PlanPurchase'] = cdf['PlanPurchase'].fillna(0).apply(lambda x: 1 if x > 0 else 0)
    for column in ['Education', 'Occupation', 'Income']:
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

    print(X.columns)
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the scaled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train a Logistic Regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    # print(result.summary())
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return model, scaler, label_encoders

# Function to predict PlanPurchase based on input data
def predict(input_data, model, scaler, label_encoders):
    required_fields = ['NoofLogin', 'ProfileViews', 'InterestSent', 'SalesTeamCallMade',
                       'Gender', 'Age', 'City', 'Income', 'Education', 
                       'Occupation', 'profilecreatedBy', 'maritalStatus', 'Height']
    
    # Check if all required fields are present
    for field in required_fields:
        if field not in input_data:
            return {'error': 'Please fill all the input fields'}

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

        # Convert prediction to boolean
        result = bool(predicted_plan_purchase)
        
        return {'PlanPurchase': result}
    
    except Exception as e:
        return {'error': str(e)}

if __name__ == '__main__':
    # Load the model and preprocess data
    model, scaler, label_encoders = load_and_train_model()

    # Example input data (replace with actual input for real prediction)
    input_data = {
        'NoofLogin': 5,
        'ProfileViews': 3,
        'InterestSent': 1,
        'SalesTeamCallMade': 2,
        'Gender': 'Male',
        'Age': 30,
        'City': 'New York',
        'Income': 'High',
        'Education': 'Masters',
        'Occupation': 'Engineer',
        'profilecreatedBy': 'System',
        'maritalStatus': 'Single',
        'Height': 175
    }

    # Get prediction
    # result = predict(input_data, model, scaler, label_encoders)
    cname = "maritalStatus"
    cindex = 11
    ar = np.array([[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[0.27],[1]]).transpose()
    invval1 = scaler.inverse_transform(ar)
    profilecreatedBy = int(invval1[0][cindex])
    print(profilecreatedBy)
    print(label_encoders[cname].inverse_transform(np.array([[profilecreatedBy]])))
    # print(result)
    # print(ar.shape)