import os
import pandas as pd
import sys

sys.path.append(os.path.abspath('src'))
from utility.utility import load_params

# # Load parameters from params.yml
# def load_params():
#     with open("params.yaml") as f:
#         params = yaml.safe_load(f)
#     return params

def load_data(output_path):
    # Load the data source URL from params.yml
    params = load_params()
    DATA_SOURCE_URL = params['data_source']['url']

    # Convert the data into a DataFrame
    data = pd.read_csv(DATA_SOURCE_URL)

    # Save the data to the output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    load_data("data/raw/data.csv")
