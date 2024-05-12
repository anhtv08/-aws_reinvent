import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate some sample transaction data
num_transactions = 10000

# Generate a date range for 10,000 records
start_date = datetime.now() - timedelta(days=num_transactions)
end_date = datetime.now()
print(start_date)
print(end_date)
transaction_dates = pd.date_range(start=start_date, end=end_date, freq='D')

# transaction_dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
#                                  end=datetime.now(), 
#                                  freq='D', periods=num_transactions)
# print(transaction_dates.)
transaction_amounts = np.random.normal(100, 50, num_transactions)
is_fraud = np.random.binomial(1, 0.05, num_transactions)

# Create the DataFrame
data = {
    # 'transaction_date': transaction_dates,
    'transaction_amount': transaction_amounts,
    'is_fraud': is_fraud
}
df = pd.DataFrame(data)

# Save the data to a CSV file in an S3 bucket
print(df.head)
#df.to_csv('s3://your-s3-bucket/fraud-detection-training-data/training_data.csv', index=False)
