import os
# run data clean script
def run(IMAGE_DIR_LIST, CAPTION_DIR_LIST):
    '''
    :param IMAGE_DIR_LIST:
    :param CAPTION_DIR_LIST:
    :return:
    '''
    # specify the path to library clean_data.py
    CODE_PATH = "library"

    for i in range(len(IMAGE_DIR_LIST)):
        IMAGE_DIR_LIST[i] = os.path.abspath(IMAGE_DIR_LIST[i])

    IMAGE_DATA_STR = ""
    for data in IMAGE_DIR_LIST:
        IMAGE_DATA_STR += data + " "

    for i in range(len(IMAGE_DIR_LIST)):
        IMAGE_DIR_LIST[i] = os.path.abspath(IMAGE_DIR_LIST[i])

    CAPTION_DIR_STR = ""
    for data in CAPTION_DIR_LIST:
        CAPTION_DIR_STR += data + " "

    # Run data validation
    os.system(
        "python " + CODE_PATH + "/validate_data.py "  "--image_dir_list " + IMAGE_DATA_STR +
    " --caption_dir_list "+ CAPTION_DIR_STR)
