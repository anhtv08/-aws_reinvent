import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples to generate
num_samples = 1000

# Generate synthetic data
data = {
    'TransactionID': range(1, num_samples + 1),
    'Amount': np.random.randint(10, 5000, num_samples),
    'TransactionDate': pd.date_range(start='2021-01-01', periods=num_samples, freq='T'),
    'CustomerID': np.random.randint(1000, 1100, num_samples),
    'MerchantID': np.random.randint(5000, 5100, num_samples),
    'Category': np.random.choice(['groceries', 'electronics', 'apparel', 'dining', 'travel'], num_samples),
    'PreviousFraudReported': np.random.choice([0, 1], num_samples, p=[0.9, 0.1]),
    'IsFraud': np.random.choice([0, 1], num_samples, p=[0.95, 0.05])
}

# Create DataFrame
df = pd.DataFrame(data)

# Save DataFrame to CSV
df.to_csv('data/fraud-detection-data.csv', index=False)

print("CSV file 'fraud-detection-data.csv' has been created and saved.")

