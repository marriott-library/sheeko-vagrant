import os
import json
from os import listdir
from os.path import isfile, join
from library import detect_run as app
# --------------- configuration start here --------------------------------------------------------------------------
# Path to checkpoint file or a directory containing checkpoint files. Passing
# a directory will only work if there is also a file named 'checkpoint' which
# lists the available checkpoints in the directory. It will not work if you
# point to a directory with just a copy of a model checkpoint: in that case,
# you will need to pass the checkpoint path explicitly.

# Path to pretrained model graph.pb file
CHECKPOINT_PATH = "path/to/dir/pretrained_model/graph.pb"


# Path to label mapping pbtxt file
VOCAB_FILE = "path/to/dir/pretrained_model/label_map.pbtxt"


# List of paths to JPEG image files to caption. It's no longer needed since image_dir is used to grab all images inside to generate the captions, for legacy testing, please go to research/im2tx test_model.py
IMAGE_DIR_LIST = ["path/to/dir/"]

# Uncomment the following section if you want the script to traverse all child dirs under a given dir. 
'''
# Example of looping all dirs under given path
DATA_DIR = '/path/to/the_master_dir'

for dir in os.listdir(os.path.abspath(DATA_DIR)):
    if os.path.isdir(os.path.abspath(os.path.join(os.path.abspath(DATA_DIR),dir))):
        DATA_DIR_LIST.append(DATA_DIR + '/' + dir)
'''

# Path to output json file
OUTPUT_PATH = "path/to/output/object_detect.json"
# -------------------configuration end here-----------------------------------------------------------------------

# Run inference to generate labels
app.run(CHECKPOINT_PATH, VOCAB_FILE, IMAGE_DIR_LIST, OUTPUT_PATH)

print("Object Detection Label Generating Completed!")
press = raw_input("Press any key to quit \n")



