import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('MetaData-Modfied.csv')

# Select features and target variable
X = data[
    [
        'Lessthnhighschool18to24', 'Highschoolgraduate18to24',
        'Somecollegeorassociatedegree18to24', 'Bachelordegreeorhigher18to24',
        'Lessthan9thgrade', '9thto12thgradenodiploma', 'Highschoolgraduate',
        'Somecollegenodegree', 'Associatedegree', 'Bachelordegree',
        'Graduateprofessionaldegree', 'Highschoolgraduatehigher',
        'Totalhousingunits', 'UnemploymentrateofPopulation16yearsandover',
        'Hispanic', 'White', 'Black', 'AmericanIndianandAlaskaNativealone',
        'Asianalone', 'NativeHawaiian', 'other', 'Under5years', '5to9years',
        '10to14years', '15to19years', '20to24years', '25to34years',
        '35to44years', '45to54years', '55to59years', '60to64years',
        '65to74years', '75to84years', '85yearsandover', 'Medianage'
    ]
]
y = data['TotalCase']
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

# Handle missing data
X = X.replace({',': ''}, regex=True)
X = X.replace('-', np.nan)
X = X.replace('3500+', 3500)
X = X.astype(float)
X.fillna(X.mean(), inplace=True)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define regression models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Fit models and evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "R2": r2}

# Print results
print("Model Performance:")
for model_name, metrics in results.items():
    print(f"{model_name}: Mean Squared Error = {metrics['MSE']:.2f}, R2 Score = {metrics['R2']:.4f}")
