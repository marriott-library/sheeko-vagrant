import os
# run data build script
def run(IMAGE_DIR_LIST, CAPTION_DIR_LIST, OUTPUT_PATH, FIELD_NM, FIELD_ID,SEG_METHOD, TRAIN_PERCENT):
    '''
    :param IMAGE_DIR_LIST: list, List of paths to directories containing JPEG image files
    :param CAPTION_DIR_LIST: list, List of paths to directories containing annotation files
    :param OUTPUT_PATH: str, Path to output directory
    :param FIELD_NM: str, Field name in annotation file containing metadata
    :param FIELD_ID: str, Field name in annotation file containing image file name
    :param SEG_METHOD: str, seg_by_image or seg_by_dir
    :param TRAIN_PERCENT: int, percentage of training in the data set
    :return:
    '''
    # specify the path to library build_data.py
    CODE_PATH = "library"

    for i in range(len(IMAGE_DIR_LIST)):
        IMAGE_DIR_LIST[i] = os.path.abspath(IMAGE_DIR_LIST[i])

    IMAGE_DATA_STR = ""
    for data in IMAGE_DIR_LIST:
        IMAGE_DATA_STR += data + " "

    for i in range(len(CAPTION_DIR_LIST)):
        CAPTION_DIR_LIST[i] = os.path.abspath(CAPTION_DIR_LIST[i])

    CAPTION_DATA_STR = ""
    for data in CAPTION_DIR_LIST:
        CAPTION_DATA_STR += data + " "

    OUTPUT_PATH = os.path.abspath(OUTPUT_PATH)

    TRAIN_PERCENT = str(TRAIN_PERCENT)

    # Run data build.
    os.system(
        "python " + CODE_PATH + "/build_data.py " + " --metadata_field=" + FIELD_NM  + ' --field_id=' + FIELD_ID+ " --output_path=" + OUTPUT_PATH + " --seg_method=" + SEG_METHOD + " --training_percent=" + TRAIN_PERCENT + " --image_dir_list " + IMAGE_DATA_STR +
    " --caption_dir_list "+ CAPTION_DATA_STR)

