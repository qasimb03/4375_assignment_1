'''
Qasim Bhutta & Camden Alpert 
Assignment 1
Anurag Nagar
CS 4375
'''

# Import necessary packages
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo 

# Fetch dataset 
obesity_level = fetch_ucirepo(id=544)


# Initialize LabelEncoder
le = LabelEncoder()

# Initialize Standard Scaler
scaler = StandardScaler()
  
# Data (as pandas dataframes);
features_df = obesity_level.data.features
targets_df = obesity_level.data.targets

# Combined DataFrame (features + target)
df = obesity_level.data.original

### Data PreProcessing ###
#Drops duplicate rows from all data DF
df.drop_duplicates(inplace=True)

# Drops NaN from all data DF
df.dropna(axis=1, inplace=True)

# Identifies categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Identifies numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Apply Label Encoding to each categorical column
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Scales numeric columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Assigns features_df to encoded and scaled features data
features_df = df.drop(columns=['NObeyesdad'])

# Assigns features_df to encoded target
targets_df = df['NObeyesdad']

# Print correlation to see which features to include/remove for learning
print(df.corr())

# Which features to include in learning
features_df = df[['Age', 'Weight', 'family_history_with_overweight', 'CAEC']]

# 90/10 split
x_train, x_test, y_train, y_test = train_test_split(features_df, targets_df, test_size=0.1, random_state=10)

# Train Model. SGDRegressor uses Stochastic Gradient Descent method
model = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='adaptive', eta0=0.001, random_state=10)
model.fit(x_train, y_train)

### Evaluation ###
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print(f'Learned Coefficients: {model.coef_}')

print(f'R-squared (Train): {r2_train}')
print(f'R-squared (Test): {r2_test}')

print(f'MSE (Train): {mse_train}')
print(f'MSE (Test): {mse_test}')