import boto3
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

# Set up the AWS session and SageMaker client
session = boto3.Session()
sagemaker = session.client('sagemaker')

# Define the training job parameters
job_name = 'my-sagemaker-training-job'
image_uri = 'your-ecr-container-image-uri'
instance_type = 'ml.m5.large'
role_arn = 'your-sagemaker-execution-role-arn'

# Create the SageMaker estimator
estimator = Estimator(
    image_uri=image_uri,
    instance_type=instance_type,
    instance_count=1,
    role=role_arn
)

# Set the training data location
s3_input_data = 's3://your-s3-bucket/training-data/'
training_input = TrainingInput(s3_input_data, content_type='text/csv')

# Start the training job
estimator.fit({'training': training_input}, job_name=job_name)
