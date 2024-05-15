import pytest
import pandas as pd
from inference import load_model, preprocess_data, predict

# Load test data
test_data = pd.read_csv("ai_machine_learning/data_preparation/data/test_data.csv")

def test_load_model():
    model_path = "path/to/model.pkl"
    model = load_model(model_path)
    assert model is not None

def test_preprocess_data():
    preprocessed_data = preprocess_data(test_data)
    assert preprocessed_data.shape[1] > test_data.shape[1]

def test_predict():
    model_path = "path/to/model.pkl"
    model = load_model(model_path)
    preprocessed_data = preprocess_data(test_data)
    predictions = predict(model, preprocessed_data)
    assert len(predictions) == len(test_data)
