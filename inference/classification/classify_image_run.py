import os
import json
from os import listdir
from os.path import isfile, join
from library import classify_run as app
# --------------- configuration start here --------------------------------------------------------------------------
# Path to checkpoint file or a directory containing checkpoint files. Passing
# a directory will only work if there is also a file named 'checkpoint' which
# lists the available checkpoints in the directory. It will not work if you
# point to a directory with just a copy of a model checkpoint: in that case,
# you will need to pass the checkpoint path explicitly.
# Path to classify_image_graph_def.pb
CHECKPOINT_PATH = "../../pretrained_models/labels/classification/classify_image_graph_def.pb"

# Path to vocabulary directory that containing pbtxt and txt dictionary file.
VOCAB_DIR = "../../pretrained_models/labels/classification/"

# List of paths to JPEG image file to caption. It's no longer needed since image_dir is used to grab all images inside to generate the captions, for legacy testing, please go to research/im2tx test_model.py
IMAGE_DIR_LIST = ["../test/"]

'''
# Example of looping all dirs under given path
IMAGE_DIR_LIST = []
DATA_DIR = 'caption'

for dir in os.listdir(os.path.abspath(DATA_DIR)):
    if os.path.isdir(os.path.abspath(os.path.join(os.path.abspath(DATA_DIR),dir))):
        DATA_DIR_LIST.append(DATA_DIR + '/' + dir)
'''


# specify destination of path to output json file
OUTPUT_PATH = "../test/classifications.json"

# -------------------configuration end here-----------------------------------------------------------------------

# Run inference to generate captions.
app.run(CHECKPOINT_PATH, VOCAB_DIR, IMAGE_DIR_LIST, OUTPUT_PATH)

print("Classified Label Generating Completed!")
press = raw_input("Press any key to quit \n")



