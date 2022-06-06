import json
from typing import Dict
import pandas as pd
from preprocessing import preprocessing


def main(config_path: str="config.json") -> None:
    # Load config file  
    config = Dict
    with open(config_path, "r") as json_file:
        config = json.load(json_file)
        print("Loaded config file")
    
    # Load data paths and preprocess data
    data = config["data"]
    data_root = config["settings"]["data_root"]
    preprocessing(data_root, data) 

if __name__ == "__main__":
    main()
