import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor  # Regressor instead of Classifier
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import pickle
import optuna
# from optuna.integration import SklearnObjective

# Load data
final_df = pd.read_csv('/home/abp/Documents/ModelSales/Interest Propensity/final_df1.csv')
final_df = final_df.drop(columns=['SourceAwid',	'TargetAwid'])

# List of categorical columns
categorical_cols = ['Gender_Source', 'City_Source', 'Occupation_Source',  'Income_Source', 'HighestEducation_Source',
                    'ProfileCreatedBy_Source', 'MaritalStatus_Source', 'PhysicalAppearance_Source',
                    'Complexion_Source', 'LivingHouseType_Source', 'familyType_Source', 'Caste_Source', 'Mangalik_Source',
                    'Gender_Target', 'City_Target', 'Occupation_Target', 'Income_Target', 'HighestEducation_Target',
                    'ProfileCreatedBy_Target', 'MaritalStatus_Target', 'PhysicalAppearance_Target',
                    'Complexion_Target', 'LivingHouseType_Target', 'familyType_Target', 'Caste_Target', 'Mangalik_Target']

# List of numerical columns
numerical_cols = ['Age_Source', 'Height_Source', 'NoOfPhotosAdded_Source',
                  'Age_Target', 'Height_Target', 'NoOfPhotosAdded_Target']

scaler = StandardScaler()

# Column transformer for both numerical and categorical columns
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

# Define the model
model = GradientBoostingRegressor(random_state=42)

# Function to define the objective for Optuna optimization
def objective(trial):
    # Define hyperparameters to optimize
    param_grid = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0)
    }
    
    # Create a pipeline with preprocessor and model
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(**param_grid, random_state=42))
    ])
    
    # Train the model
    pipe.fit(X_train, y_train)
    
    # Predict and calculate the mean squared error
    y_pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return mse  # Return the objective value (lower is better)

# Create an Optuna study for hyperparameter optimization
study = optuna.create_study(direction='minimize')  # Minimize MSE
study.optimize(objective, n_trials=50)  # Perform 50 trials of optimization

# Print best hyperparameters
print("Best hyperparameters:", study.best_params)

# Retrain the model with the best hyperparameters
best_params = study.best_params
best_model = GradientBoostingRegressor(**best_params, random_state=42)

# Create a final pipeline with the best hyperparameters
final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', best_model)
])

# Train the final model
final_model.fit(X_train, y_train)

# Predict and evaluate the final model
y_pred = final_model.predict(X_test)

# Evaluation metrics for regression
print("Final Model Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Final Model R-squared:", r2_score(y_test, y_pred))

# Save the final model to a file
joblib.dump(final_model, 'optuna.pkl')
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)