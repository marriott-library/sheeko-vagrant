from __future__ import unicode_literals
import str_nlp
import os
import glob
import json
import argparse
from os.path import isfile, join
'''
    Steps:
    # Remove proper-noun noise in data set during NLP process using spaCy
    # Save cleaned annotation file in the destination directory following the same directory structure
    # Create data report in JSON format under destination directory
'''
saved_dir = ""
data_dir = ""
field_name = ""


#   function to load the specified field in json object
def load_field(json_object, field_nm):
    '''
    load metadata from the json object with the specified field name
    :param json_object: json object of single collection entity
    :param field_nm:  field name to extract the metadata in the json object
    :return: str, metadata of the given field in the json object
    '''
    if field_nm in json_object:
        if not isinstance(json_object[field_nm], str) and not isinstance(json_object[field_nm], unicode):
            return ""
        elif isinstance(json_object[field_nm], str) or isinstance(json_object[field_nm], unicode):
            return json_object[field_nm]
    else:
        return ""


#   function to convert given collection list's field with specified process name
def iterate_list(data_dir, coll_nm_list, field_nm, saved_dir, step_fn):
    '''
    Process the given list of collection swith nlp script and create json files for each image
    :param data_dir: path to home directory of collections
    :param coll_nm_list: list of collection names
    :param field_nm: field to process the metadata
    :param saved_dir: output directory of the json file
    :param step_fn: single function for the iteration
    :return: int, total number of files created
    '''
    result_list = []
    if len(coll_nm_list) == 0:
        #   loop all collections under data dir
        os.chdir(data_dir)
        coll_nm_list = os.listdir(data_dir)

    if step_fn == nlp_collection:
        for coll_nm in coll_nm_list:
            result_list.append({"coll name": coll_nm, "data_count": step_fn(data_dir, coll_nm, field_nm, saved_dir)})
    elif step_fn == report_collection:
        for coll_nm in coll_nm_list:
            data_count, valid_count, invalid_count = step_fn(data_dir, coll_nm, field_nm, saved_dir)
            result_list.append({"coll name": coll_nm, "data count": data_count, "none pronoun": valid_count, "pronoun": invalid_count})
    return result_list


#   function to process both report and nlp process with the given collections list
def full_process(file_names, field_nm, saved_dir):

    result_list = []
    dir_nm_list = []
    for file in file_names:
        if os.path.dirname(file) not in dir_nm_list:
            dir_nm_list.append(os.path.dirname(file))
    for dir_nm in dir_nm_list:
        valid_data_count, data_count, valid_count, invalid_count = nlp_dir(dir_nm, field_nm, saved_dir)
        result_list.append({"dir name": dir_nm, "data count": data_count, "none pronoun": valid_count, "pronoun": invalid_count, "valid metadata": valid_data_count})
    return result_list


#   function to process single dir name
def nlp_dir(data_dir, field_nm, saved_dir):
    '''
    Process single collection with nlp script and create json files for each image
    :param data_dir: path to directory of caption annotation
    :param field_nm: field used to process nlp
    :param saved_dir: output directory to save json file
    :return: int, total number of files processes
    '''
    #   get all json file in the directory
    os.chdir(data_dir)
    valid_data_count = 0
    data_count = 0
    valid_count = 0
    invalid_count = 0
    for file in glob.glob("*.json"):
        data_count += 1
        image_id = (os.path.splitext(os.path.basename(file))[0])
        print("processing "+image_id)
        metadata_str, total_ents, total_noun_chunks = nlp_image(data_dir, image_id, field_nm, saved_dir)
        is_valid = validate_image(data_dir, image_id, field_nm)
        if is_valid:
            valid_count += 1
        # abort the image without the field specified or get "" return from nlp process
        else:
            invalid_count += 1
        if len(metadata_str)>0:
            print(image_id+" has been processed")
            valid_data_count += 1
        # abort the image without the field specified or get "" return from nlp process
        else:
            print(image_id+" get aborted")

    return valid_data_count, data_count ,valid_count, invalid_count


#   function to process single image
def nlp_image(data_dir, image_id, field_nm, saved_dir):
    '''
    Process single image with nlp script and create json file for the image
    :param data_dir: path to directory of annotations
    :param image_id: image id in the collection folder
    :param field_nm: field used to process nlp
    :param saved_dir: output directory to save json file
    :return: str, output str
    '''
#   get the path to the image's json file and load the json file
    image_id = str(image_id)
    file_nm = data_dir + "/" + image_id + ".json"

    meta_data = {}
    with open(file_nm, "r") as f:
        data = f.read()

    meta_data['image_id'] = image_id
    # load the str from the json object
    json_object = json.loads(data)
    # nlp the str
    metadata_str = load_field(json_object, field_nm)
    total_ents = 0
    total_noun_chunks = 0
    if len(metadata_str) > 0:
        metadata_str, total_ents, total_noun_chunks = str_nlp.process(metadata_str)
        while len(str_nlp.get_ents(metadata_str)['ents']) > 0:
            metadata_str, total_ents, total_noun_chunks = str_nlp.process(metadata_str)
        meta_data['caption'] = metadata_str
        # if nlp result is not null, then save the metadata to json file, otherwise abort
        if len(metadata_str) > 0:
            # create dir if data dir does not exist
            data_path = saved_dir + "/" + os.path.basename(data_dir)
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            # create new json file with the image id and metadata named (image_id)_metadata.json
            with open(data_path + '/' + image_id + ".json", 'w') as f:
                f.write(json.dumps(meta_data))
    return metadata_str, total_ents, total_noun_chunks


#   function to get report of the image
def validate_image(data_dir, image_id, field_nm):
    '''
    :param data_dir: path to home directory of images
    :param image_id: image id in the directory
    :param field_nm: field used to process nlp
    :return: str, output str
    '''
    image_id = str(image_id)
    file_nm = data_dir + "/" + image_id + ".json"
    with open(file_nm, "r") as f:
        data = f.read()
    # load the str from the json object
    json_object = json.loads(data)
    # nlp the str
    metadata_str = load_field(json_object, field_nm)
    is_valid = False
    if len(metadata_str) > 0:
        total_ents, total_noun_chunks = str_nlp.get_report(metadata_str)
        if total_ents == 0 and total_noun_chunks != 0:
            is_valid = True
    return is_valid


def main():
    global saved_dir, data_dir, field_name
    # parse args
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--output_path',
        type=str,
        help="""\
               Path to directory of output cleaned data
               """
    )

    parser.add_argument(
        '--metadata_field',
        type=str,
        default = 'description_t',
        help="""\
                Field name of metadata in annotation file
                """
    )
    parser.add_argument(
        '--data_dir_list',
        nargs='*',
        dest='data_dir_list'
    )

    args, unparsed = parser.parse_known_args()
    data_dir_list = args.data_dir_list
    file_names = []
    for dir in data_dir_list:
        for file in os.listdir(dir):
            if isfile(join(dir, file)) and file.endswith('json'):
                file_names.append(join(dir, file))

    saved_dir = args.output_path
    field_name = args.metadata_field
    result = full_process(file_names, field_name, saved_dir)
    if not (os.path.isdir(saved_dir)):
        os.mkdir(saved_dir)
    with open(saved_dir + '/data_report.json', 'w') as f:
        f.write(json.dumps(result))
    print("data cleaning finished!, %d directories in total processed." % (len(result)))


if __name__ == "__main__":
    main()

