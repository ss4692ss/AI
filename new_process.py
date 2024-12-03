import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor

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
        '65to74years', '75to84years', '85yearsandover', 'Medianage', 'Distance_To_Police'
    ]
]
y = data['TotalCase']
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()


X = X.replace({',': ''}, regex=True)
X = X.replace('-', np.nan)
X = X.replace('3500+', 3500)
X = X.astype(float)
X.fillna(X.mean(), inplace=True)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the model function
def build_model():
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Train the model using KerasRegressor
regressor = KerasRegressor(model=build_model, epochs=100, batch_size=32, verbose=1)
regressor.fit(X_train, y_train)

# Evaluate the model
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"ANN Model - MSE: {mse}, MAE: {mae}, R²: {r2}")

# Feature Importance using Permutation Importance
perm_importance = permutation_importance(regressor, X_test, y_test, scoring='neg_mean_squared_error', n_repeats=10, random_state=42)
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(feature_importances)

# Plot training history
plt.figure(figsize=(12, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()

# Save the inner Sequential model
regressor.model_.save("ann_model.h5")

