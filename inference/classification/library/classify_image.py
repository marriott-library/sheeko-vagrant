# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs object_detect on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

import json
from os import listdir
from os.path import isfile, join

# reads it back
# decoding the JSON to dictionay

caption_object = []

class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.vocab_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.vocab_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(FLAGS.checkpoint_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image, index):
  """Runs object_detect on an image.

  Args:
    image: Image file name.

  Returns:
    Nothing
  """
  image_object={}
  image_object["file_name"]=os.path.basename(image)
  image_object["id"] = index
  image_object["collection_name"] = ""
  image_object["annotation"] = {"labels": [], "captions": []}
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    # TODO: make it a loop
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()

    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    i = 0
    labels_num = 0
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      #print('%s (score = %.5f)' % (human_string, score))
      label_texts = human_string.split(",")
      labels_num+=len(label_texts)  
      image_object["annotation"]["labels"].append({"label_texts" : label_texts, "score" : str(score), "first_class" : False})    
      if labels_num<=3:
        image_object["annotation"]["labels"][i]["first_class"] = True     
      elif labels_num>3 and (labels_num - len(label_texts) < 3):
        image_object["annotation"]["labels"][i]["first_class"] = True
      i += 1
    caption_object["images"].append(image_object)


def run_inference_on_images(images):
  """
  Runs object_detect on given images.

  Args:
    images: Image file names.

  Returns:
    Nothing
  """
  index = 0
  # Creates graph from saved GraphDef.
  create_graph()
  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    file_num = 0
    for image in images:
      print("handling %s" % os.path.basename(image))
      image_object = {}
      image_object["file_name"] = os.path.basename(image)
      image_object["labels"] = []
      if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
      image_data = tf.gfile.FastGFile(image, 'rb').read()
      softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
      predictions = sess.run(softmax_tensor,
                             {'DecodeJpeg/contents:0': image_data})
      predictions = np.squeeze(predictions)
      # Creates node ID --> English string lookup.
      node_lookup = NodeLookup()

      top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
      i = 0
      labels_num = 0
      for node_id in top_k:
        human_string = node_lookup.id_to_string(node_id)
        score = predictions[node_id]
        # print('%s (score = %.5f)' % (human_string, score))
        label_texts = human_string.split(",")
        labels_num += len(label_texts)
        image_object["labels"].append(
          {"label_texts": label_texts, "score": str(score), "first_class": False})
        if labels_num <= 3:
          image_object["labels"][i]["first_class"] = True
          print(label_texts)
        elif labels_num > 3 and (labels_num - len(label_texts) < 3):
          image_object["labels"][i]["first_class"] = True
          print(label_texts)
        i += 1
      caption_object.append(image_object)
      print("%s has been process" % os.path.basename(image))
      file_num +=1
      if file_num == 20:
        print('saving caption objects...')
        with open(FLAGS.output_path, 'w') as f:
          data = json.dumps(caption_object)
          f.write(data)
        file_num = 0
        print('file saved!  %d images in caption object' % len(caption_object))

      index += 1

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
  run_inference_on_images(images)
  print('saving caption objects... %d new objects added.' % (len(caption_object) - original_total_captions))
  data = json.dumps(caption_object)
  with open(FLAGS.output_path, "w") as f:
    f.write(data)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  # imagenet_synset_to_human_label_map.txt:
  #   Map from synset ID to a human readable string.
  # imagenet_2012_challenge_label_map_proto.pbtxt:
  #   Text representation of a protocol buffer mapping a label to synset ID.
  parser.add_argument(
      '--checkpoint_path',
      type=str,
      default='/nfs/lyrasis/models/label/v3/classify_image_graph_def.pb',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
    '--vocab_dir',
    type=str,
    default='/nfs/lyrasis/models/label/v3/',
    help="""\
       Path to classify_image_graph_def.pb,
       imagenet_synset_to_human_label_map.txt, and
       imagenet_2012_challenge_label_map_proto.pbtxt.\
       """
  )
  parser.add_argument(
      '--output_path',
      type=str,
      default='../output/classification.json',
      help='Absolute path to output file.'
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=5,
      help='Top 5 results'
  )
  parser.add_argument(
        '--image_dir_list',
        nargs='*',
        dest='image_dir_list'

    )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
