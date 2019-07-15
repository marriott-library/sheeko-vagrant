import json
import os
from PIL import Image
from resizeimage import resizeimage
from random import shuffle
import math
import shutil
import argparse
from os.path import isfile, join
'''
    Steps:
    # Filter data set by getting data that have both images and associating annotation only
    # Segment images and annotation files into training and testing data sets
    # Resize image to trainable format into training and testing data set
    # Build annotation file for training and testing data set
'''

# fields in json object that containing metadata and ids
field_name = ""
field_id = ""
# output path
saved_dir = ""
# set up the maximum size of image that can be processed by this script
Image.MAX_IMAGE_PIXELS = 933120000


# function to resize single image and save resized image to destination directory
def resize_image(image_path, output_dir):
    '''
    :param image_path: str, path to image
    :param output_dir: str, path to destination directory saving the resized image
    :return: boolean, indicate if image exists and is valid
    '''
    image_exist = False
    if os.path.exists(image_path):
        file_info = os.stat(image_path)
        if file_info.st_size < 15424400:
            image_exist = True
    else:
        return image_exist
    with open(image_path, 'r+b') as f:
        image = Image.open(f)
        width, height = image.size
        if width >= height:
            if width > 640:
                width = 640
            image = resizeimage.resize_width(image, width)
        else:
            if height > 640:
                height = 640
            image = resizeimage.resize_height(image, height)

            # create new resized image at output_dir
        image.save(output_dir + os.path.basename(image_path), image.format)
    return image_exist


# function to process single image's metadata
def process_image_metadata(file_name, image_dir, total_captions):
    '''
    :param file_name: str
    :param image_dir: str
    :return: two dicts, image and annotation
    '''
    with open(file_name, "r") as f:
                data = f.read()
                json_object = json.loads(data)
                annotation = {"caption": json_object[field_name], "image_id": json_object[field_id], "id": total_captions}
                image = {"id": json_object[field_id], "file_name": image_dir}
    return image, annotation


# function to segment data set in each dir with given percentage of data into training set, the rest into test
def seg_by_image(caption_file_paths, image_file_paths, training_percent):
    '''
    :param caption_file_names: list, list of paths to annotation files
    :param image_file_names: list, list of dirs of image files
    :param training_percent: int, percent of training data in the data set
    :return: train_caption_object, test_caption_object containing images and annotations
    '''

    train_caption_object = {"images": [], "annotations": []}
    test_caption_object = {"images": [], "annotations": []}

    # clean up destination directory
    if os.path.exists(saved_dir + '/test/'):
        shutil.rmtree(saved_dir + '/test')
    if os.path.exists(saved_dir + '/train/'):
        shutil.rmtree(saved_dir + '/train')

    # base name without extension as reference to caption_file_paths using index
    caption_ids = []
    # base name without extension as reference to image_file_paths using index
    image_ids = []

    # save base name in ids list matching up file paths

    for i in range(len(caption_file_paths)):
        with open(caption_file_paths[i]) as f:
            data = f.read()
            data = json.loads(data)
            caption_ids.append(data[field_id])

    for i in range(len(image_file_paths)):
        image_ids.append(os.path.basename(image_file_paths[i]).split('.')[0])

    # keep files only have both annotation and image files
    # each image may have multiple annotation files
    image_caption_pairs = []
    for i in range(len(caption_ids)):
        if caption_ids[i] in image_ids:
            image_caption_pairs.append({'annotation': i, 'image': image_ids.index(caption_ids[i])})

    image_list = []
    for item in image_caption_pairs:
        if item["image"] not in image_list:
            image_list.append(item["image"])

    image_name_list = []
    for image_index in image_list:
        image_file_name = image_file_paths[image_index]
        print("Start Processing %s" % os.path.basename(image_file_name))
        file_info = os.stat(image_file_name)

        if file_info.st_size > 15424400:
            continue

        name = os.path.basename(image_file_name).split('.')[0]
        image_name_list.append(name)

    '''
  
    # remove data those image exceeds size limit
    image_name_list = []
    for item in image_caption_pairs:
        print("Start Processing %s" % caption_ids[item['annotation']])
        image_id = image_file_paths[item['image']]
        file_info = os.stat(image_id)

        if file_info.st_size > 15424400:
            continue

        name = os.path.basename(image_id).split('.')[0]
        image_name_list.append(name)
    '''
    # separate data based on the percentage
    train_name_set, test_name_set = split_data(image_name_list, training_percent)
    train_caption_list = []
    test_caption_list = []
    train_image_list = []
    test_image_list = []

    # add file paths to list for training and testing sets
    train_set = []
    for i in range(len(train_name_set)):
        caption_indexes = [j for j, x in enumerate(caption_ids) if x == train_name_set[i]]
        train_set.append({"image": image_file_paths[image_ids.index(train_name_set[i])], "captions": [caption_file_paths[index] for index in caption_indexes]})
        train_caption_list.append(caption_file_paths[caption_ids.index(train_name_set[i])])
        train_image_list.append(image_file_paths[image_ids.index(train_name_set[i])])

    test_set = []
    for i in range(len(test_name_set)):
        caption_indexes = [j for j, x in enumerate(caption_ids) if x == test_name_set[i]]
        test_set.append({"image": image_file_paths[image_ids.index(test_name_set[i])],
                          "captions": [caption_file_paths[index] for index in caption_indexes]})
        test_caption_list.append(caption_file_paths[caption_ids.index(test_name_set[i])])
        test_image_list.append(image_file_paths[image_ids.index(test_name_set[i])])
    # pass list of paths to captions files and paths to image files,
    # caption object ,image resize destination directory to process_list function
    train_caption_object = process_list(train_set, train_caption_object,
                                        saved_dir + '/train/images/')
    test_caption_object = process_list(test_set, test_caption_object,
                                        saved_dir + '/test/images/')
    # save annotation results to destination directory
    if not os.path.exists(saved_dir + '/train/annotations/'):
        os.makedirs(saved_dir + '/train/annotations/')
    #   write annotation json file to output_dir
    with open(saved_dir + '/train/annotations/' + 'annotation.json', 'w') as f:
        data = json.dumps(train_caption_object)
        f.write(data)
    if not os.path.exists(saved_dir + '/test/annotations/'):
        os.makedirs(saved_dir + '/test/annotations/')
    #   write annotation json file to output_dir
    with open(saved_dir + '/test/annotations/' + 'annotation.json', 'w') as f:
        data = json.dumps(test_caption_object)
        f.write(data)

    return train_caption_object, test_caption_object


# function to segment dirs with given percentage into training set, the rest of dirs into test
def seg_by_dir(caption_dir_list, image_dir_list, training_percent):
    '''
    :param caption_dir_list: list, list of paths to dirs of annotation files
    :param image_dir_list: list, list of paths to dirs of image files
    :param training_percent: int, percent of training data in the data set
    :return: train_caption_object, test_caption_object
    '''

    # clean up destination directory
    if os.path.exists(saved_dir + '/test/'):
        shutil.rmtree(saved_dir + '/test')
    if os.path.exists(saved_dir + '/train/'):
        shutil.rmtree(saved_dir + '/train')

    train_caption_object = {"images": [], "annotations": []}
    test_caption_object = {"images": [], "annotations": []}

    # image list loading all images, file paths is the reference
    image_ids = []
    image_file_paths = []

    caption_ids = []
    caption_file_paths = []

    # load all images into list
    for dir in image_dir_list:
        files = os.listdir(dir)
        for file in files:
            if join(dir, file) and file.endswith('jpg'):
                image_ids.append(file.split('.')[0])
                image_file_paths.append(join(dir, file))

    # load all captions into list
    for dir in caption_dir_list:
        files = os.listdir(dir)
        for file in files:
            if join(dir, file) and file.endswith('json'):
                with open(join(dir, file), 'r') as f:
                    data = f.read()
                    data = json.loads(data)
                    caption_ids.append(data[field_id])
                caption_file_paths.append(join(dir, file))
    valid_list = []
    # get valid images only has annotation file and does not exceed file size
    for i in range(len(image_file_paths)):
        image_file = image_file_paths[i]
        file_info = os.stat(image_file)
        if file_info.st_size > 15424400 or image_ids[i] not in caption_ids:
            continue
        valid_list.append(image_ids[i])

    # split image directories into training set and testing set
    train_image_dir_set, test_image_dir_set = split_data(image_dir_list, training_percent)

    train_set = []
    test_set = []
    # add images and annotations to train and test list
    for dir in train_image_dir_set:
        files = os.listdir(dir)
        for file in files:
            if join(dir, file) and file.endswith('jpg') and file.split(".")[0] in valid_list:
                caption_indexes = [j for j, x in enumerate(caption_ids) if x == file.split(".")[0]]
                train_set.append({"image": image_file_paths[image_ids.index(file.split(".")[0])],
                                  "captions": [caption_file_paths[index] for index in caption_indexes]})

    for dir in test_image_dir_set:
        files = os.listdir(dir)
        for file in files:
            if join(dir, file) and file.endswith('jpg') and file.split(".")[0] in valid_list:
                caption_indexes = [j for j, x in enumerate(caption_ids) if x == file.split(".")[0]]
                test_set.append({"image": image_file_paths[image_ids.index(file.split(".")[0])],
                                  "captions": [caption_file_paths[index] for index in caption_indexes]})

    # pass list of paths to captions files and paths to image files, caption object,
    # image resize destination directory to process_list function
    train_caption_object = process_list(train_set, train_caption_object,
                                    saved_dir + '/train/images/')
    test_caption_object = process_list(test_set, test_caption_object,
                                   saved_dir + '/test/images/')

    if not os.path.exists(saved_dir + '/train/annotations/'):
        os.makedirs(saved_dir + '/train/annotations/')
    #   write annotation json file to output_dir
    with open(saved_dir + '/train/annotations/' + 'annotation.json', 'w') as f:
        data = json.dumps(train_caption_object)
        f.write(data)
    if not os.path.exists(saved_dir + '/test/annotations/'):
        os.makedirs(saved_dir + '/test/annotations/')
    #   write annotation json file to output_dir
    with open(saved_dir + '/test/annotations/' + 'annotation.json', 'w') as f:
        data = json.dumps(test_caption_object)
        f.write(data)
    return train_caption_object, test_caption_object


# function to shuffle the given ids and put into train and test set
def split_data(image_id_list, train_percent):
    shuffle(image_id_list)
    train_count = int(math.floor(len(image_id_list) * (train_percent * 0.01)))
    train_id_set = image_id_list[:train_count]
    test_id_set = image_id_list[train_count:]
    return train_id_set, test_id_set


# function to process the images and captions with given path
def process_list(train_pair_list, caption_object, image_output_dir):
    '''
    :param train_pair_list: list of pairs containing paths to images and captions. Each image may have multiple captions
    :param caption_object: caption_object that will store the result
    :param image_output_dir: path to destination of resize images
    :return: updated caption_object
    '''

    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)

    total_captions = 0
    #   process image and metadata file loop
    for i in range(len(train_pair_list)):
        # resize the image with id to smaller
        image_dir = train_pair_list[i]["image"]
        image_exist = resize_image(image_dir, image_output_dir)
        if not image_exist:
            continue
        for caption_dir in train_pair_list[i]["captions"]:
            total_captions += 1
            #   process the json file
            image, annotation = process_image_metadata(caption_dir, image_dir, total_captions)
            if image not in caption_object["images"]:
                caption_object["images"].append(image)
            caption_object["annotations"].append(annotation)
    return caption_object


def main():
    global field_name, field_id, saved_dir
    # parse args
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--image_dir_list',
        nargs='*',
        dest='image_dir_list',
        help="""\
        List of paths to root directory of images for data build
        """
    )

    parser.add_argument(
        '--caption_dir_list',
        nargs='*',
        dest='caption_dir_list',
        help="""\
                List of paths to root directory of annotation files for data build
                """
    )

    parser.add_argument(
        '--output_path',
        type=str,
        help="""\
            Path to directory of output formatted data 
            """
    )

    parser.add_argument(
        '--seg_method',
        type=str,
        default='seg_by_image',
        help="""\
             segment method from seg_by_image, seg_by_dir
             """
    )

    parser.add_argument(
        '--metadata_field',
        type=str,
        default='description_t',
        help="""\
                Field name of metadata in annotation file
                """
    )

    parser.add_argument(
        '--field_id',
        type=str,
        default='image_id',
        help="""\
                 Field name of image id in annotation file
                 """
    )

    parser.add_argument(
        '--training_percent',
        type=str,
        default='80',
        help="""\
                percentage in int goes into training set, the rest will go to testing set
                """
    )

    args, unparsed = parser.parse_known_args()

    image_dir_list = args.image_dir_list
    caption_dir_list = args.caption_dir_list



    saved_dir = args.output_path
    field_name = args.metadata_field
    field_id = args.field_id
    seg_method = args.seg_method
    training_percent = int(args.training_percent)

    if seg_method == "seg_by_image":
        # get all files
        caption_file_names = []
        for dir in caption_dir_list:
            for file in os.listdir(dir):
                if isfile(join(dir, file)) and file.endswith('json'):
                    caption_file_names.append(join(dir, file))

        image_file_names = []
        for dir in image_dir_list:
            for file in os.listdir(dir):
                if isfile(join(dir, file)) and file.endswith('jpg'):
                    image_file_names.append(join(dir, file))

        train_caption_object, test_caption_object = seg_by_image(caption_file_names, image_file_names, training_percent)
        print("%d images in training set, %d images om testing set have been processed" % (len(train_caption_object['images']), len(test_caption_object['images'])))
    elif seg_method == "seg_by_dir":
        train_caption_object, test_caption_object = seg_by_dir(caption_dir_list,image_dir_list, training_percent)
        print("%d images in training set, %d images om testing set have been processed" % (
        len(train_caption_object['images']), len(test_caption_object['images'])))


if __name__ == "__main__":
    main()
