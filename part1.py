'''
Qasim Bhutta & Camden Alpert 
Assignment 1
Anurag Nagar
CS 4375
'''

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Fetch dataset
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
student_performance = fetch_ucirepo(id=320)
type(student_performance)

# data (as pandas dataframes) 
x = student_performance.data.features
y = student_performance.data.targets

# variable information 
# print(student_performance.variables) 

# print(x.columns)

df = pd.DataFrame(student_performance['data']['features'])
print(df.head())
print(df.shape)


# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

# Initialize LabelEncoder
le = LabelEncoder()

# Apply Label Encoding to each categorical column
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print(df.head())

# Identify numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

scaler = StandardScaler()

# Standardize the numeric columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print(df.head())