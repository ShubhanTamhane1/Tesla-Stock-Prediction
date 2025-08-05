import os
import joblib
import pandas as pd

def model_fn(model_dir):
    """Load the model from disk"""
    model_path = os.path.join(model_dir, "svm_model.joblib")
    return joblib.load(model_path)

def input_fn(request_body, request_content_type):
    """Parse incoming data"""
    if request_content_type == "application/json":
        return pd.DataFrame([request_body])
    else:
        raise ValueError("Unsupported content type: {}".format(request_content_type))

def predict_fn(input_data, model):
    """Make a prediction"""
    return model.predict(input_data)

def output_fn(prediction, content_type):
    """Return the prediction result"""
    return str(prediction[0])
