import os
# run data clean script
def run(TRAIN_SET_IMAGE, TEST_SET_IMAGE, TRAIN_SET_CAPTION, TEST_SET_CAPTION, OUTPUT_PATH):
    '''
    :param TRAIN_SET_IMAGE:
    :param TEST_SET_IMAGE:
    :param TRAIN_SET_CAPTION:
    :param TEST_SET_CAPTION:
    :param OUTPUT_PATH:
    :return:
    '''
    # specify the path to library clean_data.py
    CODE_PATH = "library"

    TRAIN_SET_IMAGE = os.path.abspath(TRAIN_SET_IMAGE)
    TEST_SET_IMAGE = os.path.abspath(TEST_SET_IMAGE)
    TRAIN_SET_CAPTION = os.path.abspath(TRAIN_SET_CAPTION)
    TEST_SET_CAPTION = os.path.abspath(TEST_SET_CAPTION)

    OUTPUT_PATH = os.path.abspath(OUTPUT_PATH)

    # Run data cleaning.
    os.system(
        "python " + CODE_PATH + "/build_TF.py " + " --train_set_image=" + TRAIN_SET_IMAGE + " --test_set_image=" + TEST_SET_IMAGE + " --train_set_caption=" + TRAIN_SET_CAPTION + " --test_set_caption=" + TEST_SET_CAPTION + " --output_path=" + OUTPUT_PATH )

