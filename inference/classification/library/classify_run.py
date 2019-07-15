import os


def run(CHECKPOINT_PATH, VOCAB_DIR, IMAGE_DIR_LIST, OUTPUT_PATH):
    '''
    :param CHECKPOINT_PATH:
    :param VOCAB_FILE:
    :param IMAGE_DIR_LIST:
    :param OUTPUT_PATH:
    :return:
    '''
    # specify the path to library inference code
    CODE_PATH = "library"

    for i in range(len(IMAGE_DIR_LIST)):
        IMAGE_DIR_LIST[i] = os.path.abspath(IMAGE_DIR_LIST[i])

    IMAGE_DATA_STR = ""
    for data in IMAGE_DIR_LIST:
        IMAGE_DATA_STR += data + " "

    OUTPUT_PATH = os.path.abspath(OUTPUT_PATH)

    # Run inference to generate classified labels.
    os.system("python " + CODE_PATH + "/classify_image.py --checkpoint_path=" + CHECKPOINT_PATH + " --vocab_dir=" + VOCAB_DIR + " --output_path=" + OUTPUT_PATH + " --image_dir_list=" + IMAGE_DATA_STR)


