import os
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

# -------------------configuration end here-----------------------------------------------------------------------

# JPEG image file to label

import tkinter as tk
from tkinter import filedialog
from tkFileDialog import askopenfilename

root = tk.Tk()
root.withdraw()
x = askopenfilename(initialdir='../test/')

IMAGE_FILE = x

# Build the inference binary.

# Ignore GPU devices (only necessary if your GPU is currently memory
# constrained, for example, by running the training script).
# os.system('set CUDA_VISIBLE_DEVICES=""')
CODE_PATH = "library"
# Run inference to generate captions.
os.system(
    "python " + CODE_PATH + "/object_detect_demo.py --checkpoint_path=" + CHECKPOINT_PATH + " --vocab_file=" + VOCAB_FILE + " --input_file=" + IMAGE_FILE)

print("Object Detect Label Generating Completed!")
press = raw_input("Press any key to quit \n")

