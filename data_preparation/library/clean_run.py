import os
# run data clean script
def run(DATA_DIR_LIST, OUTPUT_PATH, FIELD_NM):
    '''
    
    :param DATA_DIR_LIST: list, List of paths to directories containing annotation json files
    :param OUTPUT_PATH: str, Path to output directory
    :param FIELD_NM: str, Field name in annotation file containing metadata
    :return:
    '''
    # specify the path to library clean_data.py
    CODE_PATH = "library"

    for i in range(len(DATA_DIR_LIST)):
        DATA_DIR_LIST[i] = os.path.abspath(DATA_DIR_LIST[i])

    DATA_STR = ""
    for data in DATA_DIR_LIST:
        DATA_STR += data + " "
   
    OUTPUT_PATH = os.path.abspath(OUTPUT_PATH)

    # Run data cleaning.
    os.system(
        "python " + CODE_PATH + "/clean_data.py " + " --metadata_field=" + FIELD_NM + " --output_path=" + OUTPUT_PATH + " --data_dir_list " + DATA_STR)

