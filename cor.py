import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


dataset = pd.read_csv("MetaData-Modfied.csv")
dataset = dataset.sample(frac=1).reset_index(drop=True)
columns = [
    'Lessthnhighschool18to24', 'Highschoolgraduate18to24',
    'Somecollege18to24', 'Bachelordegreeorhigher18to24',
    'Lessthan9thgrade', '9thto12thgradenodiploma', 'Highschoolgraduate',
    'Somecollegenodegree', 'Associatedegree', 'Bachelordegree',
    'Graduateprofessionaldegree', 'Highschoolgraduatehigher',
    'Totalhousingunits', 'Unemploymentrateover16',
    'Hispanic', 'White', 'Black', 'American_Indian',
    'Asianalone', 'NativeHawaiian', 'other', 'Under5years', '5to9years',
    '10to14years', '15to19years', '20to24years', '25to34years',
    '35to44years', '45to54years', '55to59years', '60to64years',
    '65to74years', '75to84years', '85yearsandover', 'Medianage',
    'Distance_To_Police', 'Total', 'Nobedroom', '1bedroom', '2bedrooms', '3bedrooms', '4bedrooms', 'Owner',
    'Animal','Dockless_vehicle','Graffiti','Historic_Preservation','Health_Sanitation','Information','Park','Property_Maintenance','Streets_Infrastructure','Solid_Waste_Services','Traffic_Signals'
]

data = dataset[columns]
data = data.replace({',': ''}, regex=True) 
data = data.replace('-', np.nan)  
data.replace('3500+', 3500, inplace=True)  
data = data.apply(pd.to_numeric, errors='coerce')  
corr_matrix = data.corr()
# plt.figure(figsize=(15, 15))
# sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
# plt.title("Correlation Matrix Heatmap")
# plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix, 
    annot=False, 
    cmap='coolwarm', 
    linewidths=0.5, 
    cbar_kws={"shrink": 0.8}, 
    xticklabels=True, 
    yticklabels=True
)

# Set smaller font sizes for labels
plt.xticks(fontsize=8, rotation=90)
plt.yticks(fontsize=8, rotation=0)

plt.title("Correlation Matrix Heatmap", fontsize=14)
plt.tight_layout()
plt.show()
