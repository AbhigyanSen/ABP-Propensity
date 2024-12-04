import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib

final_df = pd.read_csv('/home/abp/Documents/Interest/final_df1.csv')

categorical_cols = ['Gender_Source', 'City_Source', 'Occupation_Source',  'Income_Source', 'HighestEducation_Source',
                    'ProfileCreatedBy_Source', 'MaritalStatus_Source', 'PhysicalAppearance_Source',
                    'Complexion_Source', 'LivingHouseType_Source', 'familyType_Source', 'Caste_Source', 'Mangalik_Source',
                    'Gender_Target', 'City_Target', 'Occupation_Target', 'Income_Target', 'HighestEducation_Target',
                    'ProfileCreatedBy_Target', 'MaritalStatus_Target', 'PhysicalAppearance_Target',
                    'Complexion_Target', 'LivingHouseType_Target', 'familyType_Target', 'Caste_Target', 'Mangalik_Target']


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

numerical_cols = ['Age_Source', 'Height_Source', 'NoOfPhotosAdded_Source',
                  'Age_Target', 'Height_Target', 'NoOfPhotosAdded_Target']

scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', scaler, numerical_cols)
    ])

# Define target variable and features
X = final_df.drop(columns=['Flag'])
y = final_df['Flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=1, random_state=42))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print("Best Model Accuracy:", accuracy_score(y_test, y_pred_best))

# Save the model to a file
joblib.dump(best_model, 'user_like_predictor_model.pkl')