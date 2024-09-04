'''
Qasim Bhutta & Cameron [LastName]
Assignment 1
Anurag Nagar
CS 4375
'''

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Fetch dataset
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
student_performance = fetch_ucirepo(id=320) 
  
# data (as pandas dataframes) 
X = student_performance.data.features 
y = student_performance.data.targets 

print(X.columns)

# variable information 
# print(student_performance.variables) 
