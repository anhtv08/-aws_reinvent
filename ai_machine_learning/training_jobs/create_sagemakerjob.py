import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
# from boto3

# role = get_execution_role('arn:aws:iam::504441261471:role/joey_sagemaker_admin')
sagemaker_session = sagemaker.Session()
instance_type='ml.m4.xlarge'
aim_role='arn:aws:iam::504441261471:role/service-role/AmazonSageMaker-ExecutionRole-20240513T142340'
estimator = Estimator(
    image_uri=get_image_uri(boto3.Session().region_name, 'xgboost', 'latest'),
    role=aim_role,
    instance_count=1,
    instance_type=instance_type,
    output_path=f's3://joey-ml/output',
    # content_type ='csv',
    sagemaker_session=sagemaker_session
)

estimator.set_hyperparameters(max_depth=5,
                              eta=0.2,
                              gamma=4,
                              min_child_weight=6,
                              subsample=0.8,
                              silent=0,
                              objective='binary:logistic',
                              num_round=100)

bucket_name="joey-ml"
# local_data_path = 'data/fraud-detection-data.csv'
# s3_data_path = f's3://{bucket_name}/data/fraud-detection-data.csv'

# local_file_path = '/Users/anhtrang/working/re_invent/aws_training/ai_machine_learning/data_training/data/fraud-detection-data.csv'
s3_file_path = f's3://{bucket_name}/data/fraud-detection-data.libsvm'

# s3 = boto3.resource('s3')
# s3.Bucket(bucket_name).upload_file(local_file_path, s3_file_path)

data_channels = {
    'train': TrainingInput(
            s3_file_path
            # , content_type='csv'
        )
    }
estimator.fit(data_channels)

# train_data = sagemaker_session.upload_file(path='s3://{bucket_name}/data/fraud-detection-data.csv')
# data_channels = {'train': train_data}

estimator.fit(data_channels)
