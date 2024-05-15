import json
import numpy as np

# Example input data
input_data = np.array([[5.1, 3.5, 1.4, 0.2]])

# Convert the input data to JSON
input_json = json.dumps(input_data.tolist())

# Make the prediction
response = predictor.predict(input_json)
prediction = json.loads(response)

print(f"Prediction: {prediction}")