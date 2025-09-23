import os
import json
import inspect

def read_config_param(param: str):
    """
    Reads the config.json file from the caller's portfolio directory
    and returns the value for the requested parameter.
    """
    # Get the path of the calling file
    caller_file = inspect.stack()[2].filename
    caller_dir = os.path.dirname(caller_file)
    print(caller_dir)
    # Construct path to config.json
    config_path = os.path.join(caller_dir, 'config.json')

    # Check if config.json exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json found in {caller_dir}")

    # Load JSON and return requested parameter
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    if param not in config_data:
        raise KeyError(f"Parameter '{param}' not found in config.json")

    return config_data[param]