import os
from library import caption_run as app

# --------------- configuration start here --------------------------------------------------------------------------
# Location to the saved training TF files.
DATA_DIR = "/path/to/tf/files"

# Path to dictionary file generated through TF build script
VOCAB_FILE = "/path/to/vocab"

# Path to dir of the Inception checkpoint file.
INCEPTION_CHECKPOINT = "/path/to/inception_v3.ckpt"

# Directory to save or restore the training process of trained model.
MODEL_DIR = "/path/to/models"

# Number of Steps to train
TRAIN_STEPS = 100

# Select gpu device to train your model, use integer number to refer to the device: e.g. 0 -> gpu_0
GPU_DEVICE = 0

# -------------------configuration end here-----------------------------------------------------------------------


app.run(DATA_DIR, INCEPTION_CHECKPOINT, MODEL_DIR, TRAIN_STEPS, GPU_DEVICE, VOCAB_FILE)

print("Train Completed!")
press = raw_input("Press any key to quit \n")

