from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model from the .pkl file
model_path = "/home/abp/Documents/Interest/user_like_predictor_model.pkl"
best_model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the input data from the request
        data = request.get_json()

        # Convert the received data into a pandas DataFrame
        new_data = pd.DataFrame(data)

        # Make the prediction
        y_pred_new = best_model.predict(new_data)
        y_pred_proba = best_model.predict_proba(new_data)

        # Extract the probability of the predicted class (assuming binary classification)
        # For binary classification, index 1 will be the probability for the positive class
        confidence = y_pred_proba[0][1]

        # Prepare the response
        result = {
            'Predicted Flag': y_pred_new[0],
            'Confidence': confidence
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)