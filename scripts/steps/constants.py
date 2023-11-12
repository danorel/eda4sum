import pathlib


# Configurations for folder structure 

root = pathlib.Path.cwd()

MODEL_FOLDER = root / "saved_models"

# Configurations for training policies and generating pipelines

MODES = ["scattered", "concentrated"]
SAMPLING_METHODS = ["node_sampling"]
