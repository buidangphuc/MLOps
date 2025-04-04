import pandas as pd
import sys
import os
import yaml

# Load init params
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)['preprocess']

def preprocess_data(input_file, output_file):
    data = pd.read_csv(input_file)
    
    # Make preprocess path
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    data.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_data(params['input'], params['output'])