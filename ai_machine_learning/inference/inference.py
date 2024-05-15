import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the trained model
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Preprocess input data
def preprocess_data(data):
    # Assuming data is a pandas DataFrame
    numerical_cols = ['col1', 'col2']
    categorical_cols = ['col3', 'col4']

    # Scale numerical features
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse=False)
    encoded_data = encoder.fit_transform(data[categorical_cols])
    encoded_cols = encoder.get_feature_names(categorical_cols)
    data = pd.concat([data[numerical_cols], pd.DataFrame(encoded_data, columns=encoded_cols)], axis=1)

    return data

# Make predictions
def predict(model, preprocessed_data):
    predictions = model.predict(preprocessed_data)
    return predictions

# Postprocess predictions (if needed)
def postprocess_predictions(predictions):
    # Assuming you don't need any postprocessing
    return predictions
