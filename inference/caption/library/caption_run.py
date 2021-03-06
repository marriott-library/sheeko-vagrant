import os

def run(CHECKPOINT_PATH, VOCAB_FILE, IMAGE_DIR_LIST, OUTPUT_PATH):
    '''
    :param CHECKPOINT_PATH: str, Path to the directory containing the checkpoint file
    :param VOCAB_FILE: str, Path to the directory containing vocabulary dictionary file
    :param IMAGE_DIR_LIST: list, List of paths to directories containing JPEG image file to caption
    :param OUTPUT_PATH: str, Path to output json file
    :return:
    '''
    # specify the path to library inference code
    CODE_PATH = "../../library"

    for i in range(len(IMAGE_DIR_LIST)):
        IMAGE_DIR_LIST[i] = os.path.abspath(IMAGE_DIR_LIST[i])

    IMAGE_DATA_STR = ""
    for data in IMAGE_DIR_LIST:
        IMAGE_DATA_STR += data + " "

    OUTPUT_PATH = os.path.abspath(OUTPUT_PATH)

    # Run inference to generate captions.
    os.system("python " + CODE_PATH + "/run_inference.py --checkpoint_path=" + CHECKPOINT_PATH + " --vocab_file=" + VOCAB_FILE + " --output_path=" + OUTPUT_PATH + " --image_dir_list=" + IMAGE_DATA_STR)
