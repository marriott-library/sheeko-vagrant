import sys
import glob
import os
sys.path.append('.')
from library import validate_run as app
# --------------- configuration start here --------------------------------------------------------------------------

# specify list of path of dirs that contain the annotation files (could be either relative or absolute path)
CAPTION_DIR_LIST = ['test']
# specify list of path of dirs that contain the images (could be either relative or absolute path)
IMAGE_DIR_LIST = ['test']

#Uncomment the section below to traverse through the child dirs of a given dir
'''
DATA_DIR = "test/"
for dir in glob.glob(DATA_DIR+'uum_*'):
    if os.path.isdir(dir):
        CAPTION_DIR_LIST.append(dir)

for dir in glob.glob(DATA_DIR+'uum_*'):
    if os.path.isdir(dir):
        IMAGE_DIR_LIST.append(dir)
'''
# -------------------configuration end here-----------------------------------------------------------------------

app.run(IMAGE_DIR_LIST, CAPTION_DIR_LIST)

print("Data Validate Completed!")
press = raw_input("Press any key to quit \n")