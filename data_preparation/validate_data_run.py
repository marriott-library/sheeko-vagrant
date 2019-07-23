import sys
import glob
import os
sys.path.append('.')
from library import validate_run as app
# --------------- configuration start here --------------------------------------------------------------------------

# List of paths (relative or absolute) to directories containing the annotation files 
CAPTION_DIR_LIST = ['path/to/captions/']
# List of paths (relative or absolute) to directories containing the JPEG image files
IMAGE_DIR_LIST = ['path/to/images/']

#Uncomment the section below to traverse through the child dirs of a given dir
'''
DATA_DIR = "/path/to/the_master_dir"

for dir in os.listdir(os.path.abspath(DATA_DIR)):
    if os.path.isdir(os.path.abspath(os.path.join(os.path.abspath(DATA_DIR),dir))):
        CAPTION_DIR_LIST.append(DATA_DIR + '/' + dir)
        IMAGE_DIR_LIST.append(DATA_DIR + '/' + dir)  
'''
# -------------------configuration end here-----------------------------------------------------------------------

app.run(IMAGE_DIR_LIST, CAPTION_DIR_LIST)

print("Data Validate Completed!")
press = raw_input("Press any key to quit \n")