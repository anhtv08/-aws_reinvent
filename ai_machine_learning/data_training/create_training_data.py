import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate some sample transaction data
num_transactions = 10000
transaction_dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                                 end=datetime.now(), 
                                 freq='D')
transaction_amounts = np.random.normal(100, 50, num_transactions)
is_fraud = np.random.binomial(1, 0.05, num_transactions)

# Create the DataFrame
data = {
    'transaction_date': transaction_dates,
    'transaction_amount': transaction_amounts,
    'is_fraud': is_fraud
}
df = pd.DataFrame(data)

# Save the data to a CSV file in an S3 bucket
df.to_csv('s3://your-s3-bucket/fraud-detection-training-data/training_data.csv', index=False)
