import os
# run TF build script
def run(TRAIN_SET_IMAGE, TEST_SET_IMAGE, TRAIN_SET_CAPTION, TEST_SET_CAPTION, OUTPUT_PATH):
    '''
    :param TRAIN_SET_IMAGE: str, Path to directory of training set containing JPEG image files
    :param TEST_SET_IMAGE: str, Path to directory of testing set containing JPEG image files
    :param TRAIN_SET_CAPTION: str, Path to directory of training set containing annotation file
    :param TEST_SET_CAPTION: str, Path to directory of training set containing annotation file
    :param OUTPUT_PATH: str, Path to output directory
    :return:
    '''
    # specify the path to library build_TF.py
    CODE_PATH = "library"

    TRAIN_SET_IMAGE = os.path.abspath(TRAIN_SET_IMAGE)
    TEST_SET_IMAGE = os.path.abspath(TEST_SET_IMAGE)
    TRAIN_SET_CAPTION = os.path.abspath(TRAIN_SET_CAPTION)
    TEST_SET_CAPTION = os.path.abspath(TEST_SET_CAPTION)

    OUTPUT_PATH = os.path.abspath(OUTPUT_PATH)

    # Run TF build.
    os.system(
        "python " + CODE_PATH + "/build_TF.py " + " --train_set_image=" + TRAIN_SET_IMAGE + " --test_set_image=" + TEST_SET_IMAGE + " --train_set_caption=" + TRAIN_SET_CAPTION + " --test_set_caption=" + TEST_SET_CAPTION + " --output_path=" + OUTPUT_PATH )

