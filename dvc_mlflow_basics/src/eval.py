import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from urllib.parse import urlparse


os.environ['MLFLOW_TRACKING_URI']=""
os.environ['MLFLOW_TRACKING_USERNAME']=""
os.environ["MLFLOW_TRACKING_PASSWORD"]=""


# Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path, model_path=None, model_uri=None):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    # Load the model either from disk or from URI
    if model_uri:
        # Load model from MLflow model registry or artifact store
        model = mlflow.pyfunc.load_model(model_uri)
    else:
        # Load model from disk (local path)
        model = pickle.load(open(model_path, 'rb'))

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    # Log metrics to MLFLOW
    mlflow.log_metric("accuracy", accuracy)
    print(f"Model accuracy: {accuracy}")

if __name__=="__main__":
    # You can modify this to pass a model_uri parameter as needed
    evaluate(params["input"], model_path=params["output"])
