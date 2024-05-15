import boto3
import sagemaker
from sagemaker.xgboost import XGBoostModel

# Initialize the SageMaker session and role
sagemaker_session = sagemaker.Session()
role = 'arn:aws:iam::504441261471:role/service-role/AmazonSageMaker-ExecutionRole-20240513T142340'

# Define the S3 path to the model artifacts
model_artifact = 's3://joey-ml/output/model.bst'

# Create the XGBoostModel object
model = XGBoostModel(
    model_data=model_artifact,
    role=role,
    entry_point='inference.py',
    framework_version='1.2-2',
    sagemaker_session=sagemaker_session
)

# Deploy the model to an endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)