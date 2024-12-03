import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Feature importances with corrected column names
importances = model.feature_importances_
feature_importances = pd.DataFrame({
    'Feature': data[
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
    ].columns,
    'Importance': importances
})

feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(feature_importances)






