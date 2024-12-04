import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor  # Regressor instead of Classifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import joblib
import pickle

# Load data
final_df = pd.read_csv('/home/abp/Documents/Interest/final_df1.csv')

# List of categorical columns
categorical_cols = ['Gender_Source', 'City_Source', 'Occupation_Source',  'Income_Source', 'HighestEducation_Source',
                    'ProfileCreatedBy_Source', 'MaritalStatus_Source', 'PhysicalAppearance_Source',
                    'Complexion_Source', 'LivingHouseType_Source', 'familyType_Source', 'Caste_Source', 'Mangalik_Source',
                    'Gender_Target', 'City_Target', 'Occupation_Target', 'Income_Target', 'HighestEducation_Target',
                    'ProfileCreatedBy_Target', 'MaritalStatus_Target', 'PhysicalAppearance_Target',
                    'Complexion_Target', 'LivingHouseType_Target', 'familyType_Target', 'Caste_Target', 'Mangalik_Target']

# # Column transformer for preprocessing
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
#     ])

# List of numerical columns
numerical_cols = ['Age_Source', 'Height_Source', 'NoOfPhotosAdded_Source',
                  'Age_Target', 'Height_Target', 'NoOfPhotosAdded_Target']

scaler = StandardScaler()

# Updated ColumnTransformer for both numerical and categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', scaler, numerical_cols)
    ])

# Define target variable and features
X = final_df.drop(columns=['Flag'])
y = final_df['Flag']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Replace RandomForestClassifier with GradientBoostingRegressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))  # Non-linear regression model
])

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)

# Evaluation metrics for regression
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# # Grid search for hyperparameter tuning
# param_grid = {
#     'regressor__learning_rate': [0.001]
# }

# grid_search = GridSearchCV(model, param_grid, cv=5)
# grid_search.fit(X_train, y_train)

# # Best parameters from grid search
# print("Best Parameters:", grid_search.best_params_)

# # Evaluate the best model from grid search
# best_model = grid_search.best_estimator_
# y_pred_best = best_model.predict(X_test)

# Performance evaluation on the best model
print("Best Model Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Best Model R-squared:", r2_score(y_test, y_pred))

# Save the best model to a file
joblib.dump(model, 'user_like_predictor_model_regressor.pkl')
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)