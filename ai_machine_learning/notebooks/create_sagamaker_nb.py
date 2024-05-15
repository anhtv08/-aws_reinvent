import boto3

# Initialize the SageMaker client
sagemaker_client = boto3.client('sagemaker', region_name='us-east-2')

# Define the notebook instance configuration
notebook_instance_name = 'example-notebook-instance'
instance_type = 'ml.t2.medium'
role_arn = 'arn:aws:iam::504441261471:role/service-role/AmazonSageMaker-ExecutionRole-20240513T142340'

# Create the notebook instance
response = sagemaker_client.create_notebook_instance(
    NotebookInstanceName=notebook_instance_name,
    InstanceType=instance_type,
    RoleArn=role_arn
)

print(f"Notebook instance {notebook_instance_name} is being created.")