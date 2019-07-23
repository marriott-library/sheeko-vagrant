import os
from os import listdir
from os.path import isfile, join
import sys
sys.path.append('.')
from library import build_run as app

# --------------- configuration start here --------------------------------------------------------------------------
# Field name in annotation file containing metadata
FIELD_NM = "captions"
# Field name in the annotation file containing the image file name
FIELD_ID = "image_id"

# List of paths (relative or absolute) of directories containing the annotation files
CAPTION_DIR_LIST = ['path/to/captions/']
# List of paths (relative or absolute) of directories containing the images
IMAGE_DIR_LIST = ['path/to/images/']

#Uncomment the section below to traverse through the child dirs of a given dir
'''
# Example of looping all dirs under given path
DATA_DIR = '/path/to/the_master_dir'

for dir in os.listdir(os.path.abspath(DATA_DIR)):
    if os.path.isdir(os.path.abspath(os.path.join(os.path.abspath(DATA_DIR),dir))):
        CAPTION_DIR_LIST.append(DATA_DIR + '/' + dir)
        IMAGE_DIR_LIST.append(DATA_DIR + '/' + dir)     
   
'''

# Path to output directory
OUTPUT_PATH = "path/to/save/the/output"

# Segment method: seg_by_image, seg_by_dir
SEG_METHOD ='seg_by_image'

# Training set percent in int
TRAIN_PERCENT = 80

# -------------------configuration end here-----------------------------------------------------------------------

app.run(IMAGE_DIR_LIST, CAPTION_DIR_LIST, OUTPUT_PATH, FIELD_NM, FIELD_ID, SEG_METHOD, TRAIN_PERCENT)

print("Building Data Completed!")
press = raw_input("Press any key to quit \n")
