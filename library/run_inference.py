# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import tensorflow as tf

import configuration
import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary
import json
from os import listdir
from os.path import isfile, join
import argparse
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("output_path", "",
                       "Path of the output json file")


tf.logging.set_verbosity(tf.logging.INFO)


caption_object = []


def main(_):
    global caption_object
    original_total_captions = 0
    if os.path.isfile(FLAGS.output_path):
        with open(FLAGS.output_path, "r") as f:
            data = f.read()
            try:
                caption_object = json.loads(data)
                original_total_captions = len(caption_object)
                print('restoring %d objects from annotation file. ' % len(caption_object))
            except:
                print('error occurs when restoring data, no valid json data found.')

    images_exclude = []
    # retrieve data from caption object
    try:
        for image_object in caption_object:
            images_exclude.append(image_object['file_name'])
    except:
        print('no annotation file to restore')

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--image_dir_list',
        nargs='*',
        dest='image_dir_list'

    )
    args, unparsed = parser.parse_known_args()
    if not os.path.isdir(os.path.dirname(FLAGS.output_path)):
        os.mkdir(os.path.dirname(FLAGS.output_path))
    print("Start Inference Process")
    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                                   FLAGS.checkpoint_path)
    g.finalize()

    # Create the vocabulary.
    vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
    # Image folder
    filenames = []
    if len(args.image_dir_list) == 1 and args.image_dir_list[0] == "":
        print("No Image is processed because image directory provided is empty.")
        return
    for dir in args.image_dir_list:
        for f in listdir(dir):
            if isfile(join(dir, f)) and f.endswith('jpg') and f not in images_exclude:
                filenames.append(join(dir, f))

    tf.logging.info("Running caption generation on %d files matching %s",
                    len(filenames), filenames)

    with tf.Session(graph=g) as sess:
        # Load the model from checkpoint.
        restore_fn(sess)

        # Prepare the caption generator. Here we are implicitly using the default
        # beam search parameters. See caption_generator.py for a description of the
        # available beam search parameters.
        generator = caption_generator.CaptionGenerator(model, vocab)
        file_num = 0
        for filename in filenames:
            print("handling %s" % os.path.basename(filename))
            image_item = {}
            image_item["file_name"] = os.path.basename(filename)
            image_item["captions"] = []
            with tf.gfile.GFile(filename, "rb") as f:
                image = f.read()
            captions = generator.beam_search(sess, image)
            # print("Captions for image %s:" % os.path.basename(filename))
            for i, caption in enumerate(captions):
                # Ignore begin and end words.
                sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                sentence = " ".join(sentence)
                image_item["captions"].append({"caption_text": sentence, "score": math.exp(caption.logprob)})
                if i == 0:
                    print("%d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
            caption_object.append(image_item)
            print("%s has been processed!" % os.path.basename(filename))
            file_num += 1
            if file_num == 20:
                print('saving caption objects...')
                with open(FLAGS.output_path, 'w') as f:
                    data = json.dumps(caption_object)
                    f.write(data)
                file_num = 0
                print('annotation file saved!  %d images in caption object' % len(caption_object))
        print('saving caption objects... %d new objects added.' % (len(caption_object) - original_total_captions))
        data = json.dumps(caption_object)
        with open(FLAGS.output_path, "w") as f:
            f.write(data)


if __name__ == "__main__":
    tf.app.run()
