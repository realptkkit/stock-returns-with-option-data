import json
from typing import Dict
import pandas as pd
from preprocessing import preprocessing
from train import init_training
import os
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main(config_path: str = "config.json") -> None:
    # Load config file
    config = Dict
    with open(config_path, "r") as json_file:
        config = json.load(json_file)
        print("Loaded config file")
    settings = config["settings"]

    # Load data paths and preprocess data
    data = config["data"]
    data_root = settings["data_root"]

    if settings["preprocess"]:
        data_preprocessed = preprocessing(data, data_root)
        init_training(config, data_preprocessed, data_root)
    else:
        init_training(config, data, data_root)


if __name__ == "__main__":
    main()
