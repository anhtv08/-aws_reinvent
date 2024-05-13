
import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer

# Assuming you have already trained your model and the model artifacts are stored in an S3 bucket
model_data = 's3://joey-ml/output/xgboost-2024-05-13-08-58-28-874/output/model.tar.gz'
sagemaker_role = 'arn:aws:iam::504441261471:role/service-role/AmazonSageMaker-ExecutionRole-20240513T142340'

# Use a built-in SageMaker algorithm
algorithm_name = 'xgboost'


# Create a SageMaker model
model = Model(
    model_data=model_data,
    role=sagemaker_role,
    predictor_cls=algorithm_name
)

# Deploy the model to a SageMaker endpoint

# Create a predictor to make predictions
# Deploy the model to a SageMaker endpoint

predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge'
)

# Make a prediction
# response = predictor.predict(data)

