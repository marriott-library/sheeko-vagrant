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
# CHECKPOINT_PATH=""
CHECKPOINT_PATH = "../../pretrained_models/labels/object_detect/models/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12/frozen_inference_graph.pb"
CHECKPOINT_PATH = "../../pretrained_models/labels/object_detect/models/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb"
CHECKPOINT_PATH = "../../pretrained_models/labels/object_detect/models/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28/frozen_inference_graph.pb"
CHECKPOINT_PATH = "../../pretrained_models/labels/object_detect/models/faster_rcnn_resnet101_ava_v2.1_2018_04_30/frozen_inference_graph.pb"
CHECKPOINT_PATH = "../../pretrained_models/labels/object_detect/models/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28/frozen_inference_graph.pb"


# label mapping files
VOCAB_FILE = "../../pretrained_models/labels/object_detect/vocab/oid_bbox_trainable_label_map.pbtxt"
VOCAB_FILE = "../../pretrained_models/labels/object_detect/vocab/mscoco_label_map.pbtxt"
VOCAB_FILE = "../../pretrained_models/labels/object_detect/vocab/ava_label_map_v2.1.pbtxt"
VOCAB_FILE = "../../pretrained_models/labels/object_detect/vocab/oid_bbox_trainable_label_map.pbtxt"


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
OUTPUT_PATH = "../test/object_detect.json"
# -------------------configuration end here-----------------------------------------------------------------------

# Run inference to generate captions.
app.run(CHECKPOINT_PATH, VOCAB_FILE, IMAGE_DIR_LIST, OUTPUT_PATH)

print("Object Detection Label Generating Completed!")
press = raw_input("Press any key to quit \n")



