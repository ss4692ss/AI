import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import kfold_template

# Load and shuffle dataset
dataset = pd.read_csv("MetaData-Modfied.csv")
dataset = dataset.sample(frac=1).reset_index(drop=True)

# Extract target variable
target = dataset["TotalCase"].astype(float).values  # Ensure target is numeric

# Specify feature columns
columns = [
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
    '65to74years', '75to84years', '85yearsandover', 'Medianage',
    'Distance_To_Police'
]

# Extract feature data
data = dataset[columns]

# Clean data
data = data.replace({',': ''}, regex=True)  # Remove commas
data = data.replace('-', np.nan)           # Replace dashes with NaN
data = data.replace('3500+', 3500)         # Handle specific string
data = data.astype(float)                  # Convert all to float

# Handle missing values (optional: impute or drop rows/columns)
data = data.fillna(data.mean())            # Replace NaN with column means

# Scale features
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Train RandomForestClassifier with k-fold validation
model = RandomForestClassifier(criterion="gini", max_depth=2, n_estimators=100, bootstrap=True)
return_values = kfold_template.run_kfold(model, data, target, 4, True)

# Print k-fold results
average_value = sum(return_values) / len(return_values)
print("Average of return values:", average_value)

# Train on full data and compute feature importance
model.fit(data, target)
feature_importances_raw = model.feature_importances_

# Match feature names to their importances
feature_zip = zip(columns, feature_importances_raw)
feature_importances = sorted([(f, round(i, 4)) for f, i in feature_zip], key=lambda x: x[1], reverse=True)

# Display feature importances
print("Feature Importances:")
[print(f"{feature:30}: {importance}") for feature, importance in feature_importances]



# County Tracts,TRACTCE20,Animal,Dockless_vehicle,Graffiti,
# Historic_Preservation,Health_Sanitation,Information,
# Park,Property_Maintenance,Streets_Infrastructure,Solid_Waste_Services,
# Traffic_Signals,Lessthnhighschool18to24,Highschoolgraduate18to24,
# Somecollegeorassociatedegree18to24,Bachelordegreeorhigher18to24,
# Lessthan9thgrade,9thto12thgradenodiploma,Highschoolgraduate,Somecollegenodegree,
# Associatedegree,Bachelordegree,Graduateprofessionaldegree,Highschoolgraduatehigher,
# bachelordegreehigher,Totalhousingunits,UnemploymentrateofPopulation16yearsandover,
# Hispanic ,White,Black,AmericanIndianandAlaskaNativealone,Asianalone,
# NativeHawaiian ,other,Under5years,5to9years,10to14years,15to19years,
# 20to24years,25to34years,35to44years,45to54years,55to59years,
# 60to64years,65to74years,75to84years,85yearsandover,Medianage,
# Owner-occupiedhousingunitswithamortgage,Total,Nobedroom,1bedroom,2bedrooms,
# 3bedrooms,4bedrooms,5ormorebedrooms,Arrest,Distance_To_Police,Offense













