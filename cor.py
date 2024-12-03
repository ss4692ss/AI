import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
dataset = pd.read_csv("MetaData-Modfied.csv")

# Shuffle the dataset
dataset = dataset.sample(frac=1).reset_index(drop=True)

# Select the columns for analysis (same as in your previous code)
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

# Extract the relevant data columns
data = dataset[columns]

# Clean the data
data = data.replace({',': ''}, regex=True)  # Remove commas
data = data.replace('-', np.nan)  # Replace '-' with NaN
data.replace('3500+', 3500, inplace=True)  # Replace '3500+' with 3500

# Ensure that the data remains as a pandas DataFrame
data = data.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, handling errors as NaN

# Calculate the correlation matrix
corr_matrix = data.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Create the heatmap using seaborn
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)

# Show the heatmap
plt.title("Correlation Matrix Heatmap")
plt.show()