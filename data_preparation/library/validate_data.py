import os
import argparse



def main():
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

    args, unparsed = parser.parse_known_args()
    image_dir_list = args.image_dir_list
    caption_dir_list = args.caption_dir_list
    total_images = 0
    for dir in image_dir_list:
        files = os.listdir(dir)
        for file in files:
            if os.path.isfile(os.path.join(dir, file)) and file.endswith('jpg'):
                total_images += 1
    total_json = 0
    for dir in caption_dir_list:
        files = os.listdir(dir)
        for file in files:
            if os.path.isfile(os.path.join(dir, file)) and file.endswith('json'):
                total_json += 1

    print("Total number of image files in %d directories: %d" % (len(image_dir_list), total_images))
    print("Total number of json files in %d directories: %d" % (len(caption_dir_list), total_json))

    return


if __name__ == "__main__":
    main()