import json
from typing import Dict
import pandas as pd
from preprocessing import preprocessing
from train import init_training


def main(config_path: str = "config.json") -> None:
    # Load config file
    config = Dict
    with open(config_path, "r") as json_file:
        config = json.load(json_file)
        print("Loaded config file")

    # Load data paths and preprocess data
    data = config["data"]

    if config["preprocess"]:
        data_root = config["settings"]["data_root"]
        data_preprocessed = preprocessing(data, data_root)
        init_training(data_preprocessed)
    else:
        init_training(data, data_root)
    

if __name__ == "__main__":
    main()
