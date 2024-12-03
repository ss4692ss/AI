import pandas
import kfold_template
import numpy as np

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pandas.read_csv("MetaData-Modfied.csv")

dataset = dataset.sample(frac=1).reset_index()

target = dataset["Offense"].values
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

data = dataset[columns]

data = data.replace({',': ''}, regex=True)
data = data.replace('-', np.nan)  
data.replace('3500+', 3500, inplace=True)

feature_list = data.columns
data = data.values

print(feature_list)
print(target)
print(data)

machine = RandomForestClassifier(criterion="gini", max_depth=2, n_estimators=100, bootstrap = True) 
return_values = kfold_template.run_kfold(machine, data, target, 4, True)
print(return_values)


machine = RandomForestClassifier(criterion="gini", max_depth=2, n_estimators=100, bootstrap = True) 
machine.fit(data, target)
feature_importances_raw = machine.feature_importances_
print(feature_importances_raw)
print(feature_list)

feature_zip = zip(feature_list, feature_importances_raw)
print(feature_zip)

feature_importances = [ (feature, round(importance, 4)) for feature, importance in feature_zip]
feature_importances = sorted(feature_importances, key = lambda x: x[1] )
print(feature_importances)
[ print('{:14}: {}'.format(*feature_importance)) for feature_importance in feature_importances]


data = data.apply(pd.to_numeric, errors='coerce')

# Calculate the correlation matrix
corr_matrix = data.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Create the heatmap using seaborn
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Show the heatmap
plt.title("Correlation Matrix Heatmap")
plt.show()


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













