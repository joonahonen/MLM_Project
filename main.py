# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load datasets
birth_data = pd.read_csv("us_births.csv")
validation_data = pd.read_csv("us_births_test.csv")

# Data preprocessing
# Convert categorical variables to numeric
binary_columns = ['mother_diabetes_gestational', 'mother_risk_factor']
for column in binary_columns:
    birth_data[column] = birth_data[column].replace({'Y': 1, 'N': 0})
    validation_data[column] = validation_data[column].replace({'Y': 1, 'N': 0})

# Convert sex to numeric
birth_data['newborn_sex'] = birth_data['newborn_sex'].replace({'F': 0, 'M': 1})
validation_data['newborn_sex'] = validation_data['newborn_sex'].replace({'F': 0, 'M': 1})

# Define features and target
output_variable = 'newborn_birth_weight'
predictor_variables = ['month', 'mother_age', 'prenatal_care_starting_month',
                    'daily_cigarette_prepregnancy', 'daily_cigarette_trimester_1',
                    'daily_cigarette_trimester_2', 'daily_cigarette_trimester_3',
                    'mother_height', 'mother_bmi', 'mother_weight_prepregnancy',
                    'mother_weight_delivery', 'mother_diabetes_gestational',
                    'newborn_sex', 'gestation_week', 'mother_risk_factor']

X = birth_data[predictor_variables]
y = birth_data[output_variable]
X_validate = validation_data[predictor_variables]
y_validate = validation_data[output_variable]

# Scale features
standardizer = StandardScaler()
X_standardized = standardizer.fit_transform(X)
X_validate_standardized = standardizer.transform(X_validate)

# Initialize Random Forest model with cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
forest_model = RandomForestRegressor(random_state=42)

# Perform cross-validation
error_scores = -cross_val_score(forest_model, X_standardized, y, 
                             cv=kfold, scoring='neg_mean_absolute_error')
squared_error_scores = -cross_val_score(forest_model, X_standardized, y, 
                                    cv=kfold, scoring='neg_root_mean_squared_error')

print("=== Validation Results ===")
print(f"Average Error: {np.mean(error_scores):.2f} (±{np.std(error_scores):.2f})")
print(f"Average Root Squared Error: {np.mean(squared_error_scores):.2f} (±{np.std(squared_error_scores):.2f})")

# Hyperparameter tuning
tuning_parameters = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

parameter_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=tuning_parameters,
    scoring='neg_mean_absolute_error',
    cv=kfold,
    n_jobs=-1,
    verbose=1
)

parameter_search.fit(X_standardized, y)

print("\n=== Optimal Parameters ===")
print(parameter_search.best_params_)

# Train final model with best parameters
final_model = parameter_search.best_estimator_
predictions = final_model.predict(X_validate_standardized)

# Calculate final metrics
mean_abs_error = mean_absolute_error(y_validate, predictions)
root_mean_sq_error = np.sqrt(mean_squared_error(y_validate, predictions))

print("\n=== Model Performance ===")
print(f"Mean Absolute Error: {mean_abs_error:.2f}")
print(f"Root Mean Square Error: {root_mean_sq_error:.2f}")

# Feature importance plot
variable_importance = pd.DataFrame({
    'variable': predictor_variables,
    'impact': final_model.feature_importances_
})
variable_importance = variable_importance.sort_values('impact', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='impact', y='variable', data=variable_importance)
plt.title('Impact of Variables on Birth Weight')
plt.tight_layout()
plt.show()
