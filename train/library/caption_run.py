import os

def run(DATA_DIR, INCEPTION_CHECKPOINT, MODEL_DIR, TRAIN_STEPS, GPU_DEVICE = 0, VOCAB_FILE=""):
    # specify the path to library train.py
    CODE_PATH = "../library"

    DATA_DIR = os.path.abspath(DATA_DIR)
    INCEPTION_CHECKPOINT = os.path.abspath(INCEPTION_CHECKPOINT)
    MODEL_DIR = os.path.abspath(MODEL_DIR)
    VOCAB_FILE = os.path.abspath(VOCAB_FILE)
    if GPU_DEVICE == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_DEVICE

    TRAIN_STEPS = str(TRAIN_STEPS)

    # Run the training script.
    os.system(
            "python " + CODE_PATH + "/train.py --input_file_pattern=" + DATA_DIR + "/train-?????-of-00256 --inception_checkpoint_file=" + INCEPTION_CHECKPOINT + " --train_dir=" + MODEL_DIR + "/train --train_inception=false --number_of_steps=" + TRAIN_STEPS + " --vocab_file=" + VOCAB_FILE)