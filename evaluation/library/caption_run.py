import os


def run(DATA_DIR, MODEL_DIR, GPU_DEVICE=0):
    '''
    :param DATA_DIR: str, Path to the directory containing TF Records files
    :param MODEL_DIR: str, Path to directory containing model file
    :param GPU_DEVICE: int, Index of the gpu device to use for the evaluation
        Run "from tensorflow.python.client import device_lib
        device_lib.list_local_devices()"
    :return:
    '''
    # Path to library evaluate.py
    CODE_PATH = "../library"

    DATA_DIR = os.path.abspath(DATA_DIR)
    MODEL_DIR = os.path.abspath(MODEL_DIR)
    if GPU_DEVICE == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_DEVICE

    # Run the evaluation script.
    os.system(
            "python " + CODE_PATH + "/evaluate.py --input_file_pattern="+DATA_DIR+"/val-?????-of-00004 --checkpoint_dir="+MODEL_DIR+" --eval_dir="+MODEL_DIR+"/eval")


