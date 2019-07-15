import os
import json
from os import listdir
from os.path import isfile, join
import sys
sys.path.append('.')
from library import clean_run as app

# --------------- configuration start here --------------------------------------------------------------------------
# specify field name in annotation file containing metadata
FIELD_NM = "description_t"

# specify list of paths of dirs that contains the meta data that needs cleaning (could be either relative or absolute path)
# This will perform NLP cleaning and remove proper-nouns
DATA_DIR_LIST = ['']

# Uncomment the following section to traverse through all child dirs of a given dir
'''
# Example of looping all dirs under given path
DATA_DIR_LIST = []
DATA_DIR = 'caption'

for dir in os.listdir(os.path.abspath(DATA_DIR)):
    if os.path.isdir(os.path.abspath(os.path.join(os.path.abspath(DATA_DIR),dir))):
        DATA_DIR_LIST.append(DATA_DIR + '/' + dir)
'''

# specify destination of output dir
OUTPUT_PATH = ""

# -------------------configuration end here-----------------------------------------------------------------------

app.run(DATA_DIR_LIST, OUTPUT_PATH, FIELD_NM)

print("Data Cleaning Completed!")
press = raw_input("Press any key to quit \n")
