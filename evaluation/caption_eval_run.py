import os
from library import caption_run as app

# --------------- configuration start here --------------------------------------------------------------------------
# Location to saved the Evaluation TF records.
DATA_DIR = "path/to/tf/files"

# Path to the directory of model files to evaluate.
MODEL_DIR = "path/to/the/model"

# Select gpu device to train your model, use integer number to refer to the device: e.g. 0 -> gpu_0
GPU_DEVICE = 0

# -------------------configuration end here-----------------------------------------------------------------------

app.run(DATA_DIR, MODEL_DIR, GPU_DEVICE)

print("Evaluation Completed!")
press = raw_input("Press any key to quit \n")