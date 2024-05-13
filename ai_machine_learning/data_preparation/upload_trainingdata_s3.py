import boto3
import pandas as pd

# Assuming 'df' is your DataFrame containing the preprocessed data
csv_buffer = StringIO()
df.to_csv(csv_buffer, index=False)

s3_resource = boto3.resource('s3')
bucket_name = 'joey-ml'
s3_resource.Object(bucket_name, 'data/fraud-detection-data.csv').put(Body=csv_buffer.getvalue())

