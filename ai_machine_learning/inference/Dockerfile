# Base image from Amazon SageMaker XGBoost container
FROM 763104351884.dkr.ecr.us-east-2.amazonaws.com/xgboost:1.2-2

# Set the working directory
WORKDIR /opt/ml/code

# Copy the inference script to the container
COPY inference.py /opt/ml/code/inference.py

# Set the environment variables for the serving script
ENV SAGEMAKER_PROGRAM inference.py