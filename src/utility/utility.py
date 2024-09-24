import yaml

# Load parameters from params.yml
def load_params(file_path='params.yaml'):
    with open(file_path) as f:
        params = yaml.safe_load(f)
    return params