import os
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

# Path to the JPEG image file to generate label
IMAGE_FILE = "path/to/file/test_image.jpg"

# -------------------configuration end here-----------------------------------------------------------------------

CODE_PATH = "library"
# Run inference to generate labels
os.system(
    "python " + CODE_PATH + "/object_detect_demo.py --checkpoint_path=" + CHECKPOINT_PATH + " --vocab_file=" + VOCAB_FILE + " --input_file=" + IMAGE_FILE)

print("Object Detect Label Generating Completed!")
press = raw_input("Press any key to quit \n")

