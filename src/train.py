import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
import mlflow
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse

# Load configuration if available, otherwise use default values
os.environ["MLFLOW_TRACKING_URI"] = ""
os.environ["MLFLOW_TRACKING_USERNAME"] = ""
os.environ["MLFLOW_TRACKING_PASSWORD"] = ""

#Load params
with open('params.yaml') as file:
    params = yaml.safe_load(file)['train']
    # path_params = yaml.safe_load(file)['preprocess']

def hyperparameter_tuning(X_train, y_train, params):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=params, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

def train_model(params):
    # Extract only the RandomForestClassifier parameters from params
    rf_params = {
        'n_estimators': params.get('n_estimators', 100),
        'max_depth': params.get('max_depth', None),
        'random_state': params.get('random_state', 42),
        # Add other RF parameters as needed
    }
    
    # Read data from the input file, not output
    data = pd.read_csv(params['input'])
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']
    mlflow.set_experiment('diabetes')
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    with mlflow.start_run():
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=params.get('random_state', 42))
        signature = infer_signature(X_train, y_train)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
        }
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
        best_model = grid_search.best_estimator_
        
        # Train the model with the best parameters
        rf = RandomForestClassifier(**grid_search.best_params_)
        rf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        
        # Log the metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_params(grid_search.best_params_)
        
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        
        # Ensure model directory exists
        os.makedirs("model", exist_ok=True)
        
        # Save confusion matrix to file before logging it
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact("confusion_matrix.png")
        
        tracking_uri = urlparse(os.environ["MLFLOW_TRACKING_URI"])
        if tracking_uri.scheme == 'file':
            mlflow.sklearn.log_model(rf, "model", signature=signature)
        else:
            mlflow.sklearn.log_model(rf, "model", registered_model_name="diabetes_model", signature=signature)
        
        # Ensure models directory exists for saving the model per params.yaml
        os.makedirs(os.path.dirname(params['output']), exist_ok=True)
        with open(params['output'], "wb") as f:
            pickle.dump(rf, f)
        print(f"Model saved to {params['output']}")

if __name__ == "__main__":
    train_model(params)