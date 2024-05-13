import numpy as np
from sklearn.datasets import dump_svmlight_file
import pandas as pd

# Load the CSV file
df = pd.read_csv('fraud-detection-data.csv')

# Separate the features and labels
# X = df.drop('label', axis=1).values
# y = df['label'].values

# Separate features and target
# Separate the features and labels
# X = df[['Amount', 'CustomerID', 'MerchantID', 'PreviousFraudReported']].values
# y = df['IsFraud'].values

# Separate the features and labels
X = df[['Amount', 'CustomerID', 'MerchantID', 'PreviousFraudReported']].values
y = df['IsFraud'].values


# Convert to LIBSVM format
dump_svmlight_file(X, y, 'fraud-detection-data.libsvm')
