import os
from library import caption_run as app

# --------------- configuration start here --------------------------------------------------------------------------
# Path to directory containing the saved evaluate TF files
DATA_DIR = "path/to/dir/tf_files"

# Directory containing model files to evaluate.
MODEL_DIR = "path/to/the/model"

# Select gpu device to evaluate your model, use integer number to refer to the device: e.g. 0 -> gpu_0
GPU_DEVICE = 0

# -------------------configuration end here-----------------------------------------------------------------------

app.run(DATA_DIR, MODEL_DIR, GPU_DEVICE)

print("Evaluation Completed!")
press = raw_input("Press any key to quit \n")