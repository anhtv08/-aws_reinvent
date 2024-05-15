# inference.py

import os
import xgboost as xgb
import json
import numpy as np

# Define the model loading function
def model_fn(model_dir):
    """Load model from the model_dir."""
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, 'model.bst'))
    return model

# Define the input transformation function
def input_fn(request_body, request_content_type):
    """Deserialize and prepare the prediction input."""
    if request_content_type == 'application/json':
        input_data = np.array(json.loads(request_body))
        dmatrix = xgb.DMatrix(input_data)
        return dmatrix
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

# Define the prediction function
def predict_fn(input_data, model):
    """Make a prediction using the loaded model."""
    prediction = model.predict(input_data)
    return prediction

# Define the output transformation function
def output_fn(prediction, response_content_type):
    """Serialize and prepare the prediction output."""
    if response_content_type == 'application/json':
        return json.dumps(prediction.tolist())
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")