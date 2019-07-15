import os
from os import listdir
from os.path import isfile, join
import sys
sys.path.append('.')
from library import build_run as app

# --------------- configuration start here --------------------------------------------------------------------------
# specify field name that contains the metadata in the annotation file
FIELD_NM = "captions"
# specify field name that contains the image file name in the annotation file
FIELD_ID = "image_id"

# specify list of path of dirs that contain the annotation files (could be either relative or absolute path)
CAPTION_DIR_LIST = []
# specify list of path of dirs that contain the images (could be either relative or absolute path)
IMAGE_DIR_LIST = []

#Uncomment the section below to traverse through the child dirs of a given dir
'''
# Example of looping all dirs under given path
CAPTION_DIR_LIST = []
IMAGE_DIR_LIST = []
DATA_DIR = 'caption'

for dir in os.listdir(os.path.abspath(DATA_DIR)):
    if os.path.isdir(os.path.abspath(os.path.join(os.path.abspath(DATA_DIR),dir))):
        CAPTION_DIR_LIST.append(DATA_DIR + '/' + dir)
        IMAGE_DIR_LIST.append(DATA_DIR + '/' + dir)        
'''

# specify destination of output dir
OUTPUT_PATH = "path/to/output"

# specify segment method seg_by_image, seg_by_dir
SEG_METHOD ='seg_by_image'

# specify training set percent in int
TRAIN_PERCENT = 80

# -------------------configuration end here-----------------------------------------------------------------------

app.run(IMAGE_DIR_LIST, CAPTION_DIR_LIST, OUTPUT_PATH, FIELD_NM, FIELD_ID, SEG_METHOD, TRAIN_PERCENT)

print("Building Data Completed!")
press = raw_input("Press any key to quit \n")
