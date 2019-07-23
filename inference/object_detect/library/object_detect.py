import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
#   from matplotlib import pyplot as plt
from PIL import Image
from os import listdir
from os.path import isfile, join
import json
import argparse

caption_object = []

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

# This is needed to display the images.
# matplotlib inline


from object_detection.utils import label_map_util

parser = argparse.ArgumentParser()
parser.add_argument(
    '--checkpoint_path',
    type=str,
    help="Path to Directory of model used for object detection."
)
parser.add_argument(
    '--vocab_file',
    type=str,
    help="Path to label map file"
)
parser.add_argument(
    '--output_path',
    type=str,
    help='Path to Directory of the output file'
)
parser.add_argument(
    '--image_dir_list',
    nargs='*',
    dest='image_dir_list'

)
FLAGS, unparsed = parser.parse_known_args()
assert FLAGS.checkpoint_path, "--checkpoint_path is required"
assert FLAGS.vocab_file, "--vocab_file is required"
assert FLAGS.output_path, "--output_path is required"
assert FLAGS.image_dir_list, "--image_dir_list is required"


# define the model and the label mapping file to use
PATH_TO_FROZEN_GRAPH = FLAGS.checkpoint_path

PATH_TO_LABELS = FLAGS.vocab_file

detection_graph = tf.Graph()

with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


from object_detection.utils import visualization_utils as vis_util

# Size, in inches, of the output images.
IMAGE_SIZE = (4, 4)


def run_inference_for_single_image(image, sess):
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name)
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    start = time.time()
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(image, 0)})
    end = time.time()
    print("Inference Time: %s" % str(end - start))
    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def run_inference_for_all_images(filenames, graph):
    with graph.as_default():
        with tf.Session() as sess:
            file_num = 0
            # loop
            for filename in filenames:
                results = []
                labels = []
                image = Image.open(filename).convert('RGB')
                image_object = {}
                image_object["file_name"] = os.path.basename(filename)
                image_object["labels"] = []
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                output_dict = run_inference_for_single_image(image_np, sess)
                print('handling %s' % os.path.basename(filename))
                object_count = 0
                for i in range(output_dict['num_detections']):
                    try:
                        label = category_index[output_dict['detection_classes'][i]]['name']
                    except:
                        print(output_dict['num_detections'], output_dict['detection_classes'])
                        with open(FLAGS.output_path, "w") as f:
                            data = json.dumps(caption_object)
                            f.write(data)
                    score = output_dict['detection_scores'][i]

                    if label not in labels:
                        object_count += 1
                        results.append({'label_text': label, 'score': str(score)})
                        print(label)
                        labels.append(label)
                image_object["labels"] = results
                caption_object.append(image_object)
                print('%d objects found in the given image: %s' % (
                    object_count, os.path.basename(filename)))
                file_num += 1
                if file_num == 20:
                    print('saving caption objects...')
                    with open(FLAGS.output_path, 'w') as f:
                        data = json.dumps(caption_object)
                        f.write(data)
                    file_num = 0
                    print('file saved!  %d images in caption object' % len(caption_object))


def main(_):
    global caption_object
    original_total_captions = 0
    # find if file exist already
    if os.path.isfile(FLAGS.output_path):
        with open(FLAGS.output_path, "r") as f:
            data = f.read()
            try:
                caption_object = json.loads(data)
                original_total_captions = len(caption_object)
                print('restoring %d objects from annotation file. ' % len(caption_object))
            except:
                print('error occurs when restoring data, no valid json data found.')

    images = []
    images_exclude = []
    # retrieve data from caption object
    try:
        for image_object in caption_object:
            images_exclude.append(image_object['file_name'])
    except:
        print('no file to restore')
    print("Start Inference Process")
    for dir in FLAGS.image_dir_list:
        for f in listdir(dir):
            if isfile(join(dir, f)) and f.endswith('jpg') and f not in images_exclude:
                images.append(join(dir, f))
    print('%d images found in the directory' % len(images))
    run_inference_for_all_images(images, detection_graph)
    print('saving caption objects... %d new objects added.' % (len(caption_object) - original_total_captions))
    data = json.dumps(caption_object)
    with open(FLAGS.output_path, "w") as f:
        f.write(data)
        print('%d images has been processed!' % len(caption_object))


if __name__ == '__main__':
    tf.app.run()