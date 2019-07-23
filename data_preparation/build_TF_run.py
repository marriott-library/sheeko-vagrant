import os
import json
from os import listdir
from os.path import isfile, join
import sys
sys.path.append('.')
from library import TF_run as app

# --------------- configuration start here --------------------------------------------------------------------------
# Path to directory containing training set images
TRAIN_SET_IMAGE = "path/to/train/images"
# Path to directory containing testing set images
TEST_SET_IMAGE = "path/to/test/images"
# Path to training set annotation file
TRAIN_SET_CAPTION = "path/to/train/annotations.json"
# Path to testing set annotation file
TEST_SET_CAPTION = "path/to/test/annotations.json"

# Path to output directory
OUTPUT_PATH = "path/to/save/the/output"
# optional, shard number
# SHARD_NUM = 256
# -------------------configuration end here-----------------------------------------------------------------------


app.run(TRAIN_SET_IMAGE, TEST_SET_IMAGE, TRAIN_SET_CAPTION, TEST_SET_CAPTION, OUTPUT_PATH)

print("TF build Completed!")
press = raw_input("Press any key to quit \n")
