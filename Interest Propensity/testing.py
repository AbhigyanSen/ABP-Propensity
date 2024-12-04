import pandas as pd
import joblib

# Load the model from the .pkl file
model = "/home/abp/Documents/Interest/user_like_predictor_model.pkl"
best_model = joblib.load(model)

print("Testing\n")

new_data = pd.DataFrame({
    'Gender_Source': ['Male'],
    'City_Source': ['Kolkata'],
    'Occupation_Source': ['Salaried - Private'],
    'Income_Source': ['INR 20 Lacs - 30 Lacs'],
    'HighestEducation_Source': ['M.Tech'],
    'ProfileCreatedBy_Source': ['Parent'],
    'MaritalStatus_Source': ['Never Married'],
    'PhysicalAppearance_Source': ['Average'],
    'Complexion_Source': ['Wheatish'],
    'LivingHouseType_Source': ['Self-Owned Flat'],
    'familyType_Source': ['Nuclear'],
    'Caste_Source': ['Kayastha'],
    'Mangalik_Source': ['No'],
    'Gender_Target': ['Female'],
    'City_Target': ['Dum Dum'],
    'Occupation_Target': ['Salaried - Private'],
    'Income_Target': ['INR 4 Lacs - 6 Lacs'],
    'HighestEducation_Target': ['B.Tech'],
    'ProfileCreatedBy_Target': ['Parent'],
    'MaritalStatus_Target': ['Never Married'],
    'PhysicalAppearance_Target': ['Slim'],
    'Complexion_Target': ['Fair'],
    'LivingHouseType_Target': ['Parental House'],
    'familyType_Target': ['Nuclear'],
    'Caste_Target': ['Kayastha'],
    'Mangalik_Target': ['No'],
    'Age_Source': [31.0],
    'Height_Source': [5.03],
    'NoOfPhotosAdded_Source': [7],
    'Age_Target': [29.0],
    'Height_Target': [5.01],
    'NoOfPhotosAdded_Target': [8]
})

# Make sure to preprocess the new data in the same way as the training data
# Use the loaded model to predict the target (Flag)
y_pred_new = best_model.predict(new_data)

# Print the prediction
print("Predicted Flag for new data:", y_pred_new)