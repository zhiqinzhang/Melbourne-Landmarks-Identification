# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil

import numpy as np
import tensorflow as tf
# import cv2 as cv

# import settings as st
import settings as st
import time

wrong_classification_num = {"china town":0, "cooks cottage":0, "eureka tower":0,
                            "flinders station":0, "general post office":0, "melbourne central":0,
                            "parliament house":0, "royal exhibition building":0,
                            "shrine of remembrance":0, "st pauls cathedral":0, "state theatre":0,
                            "old treasury building":0, "queen victoria market":0, "town hall":0}

count = 0

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)
  # result = sess.run(resized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


def single_imgae_classifier(filename):
  file_name = filename
  # model_file = "./output/inception_v3_batch64_lr0_1/inception_v3_batch64_lr0.1.pb"
  model_file = "./output/mobilenet_v1_075_160_batch64_lr0_1/mobilenet_v1_batch64_lr0.1.pb"

  label_file = "./output/output_labels.txt"
  # input_height = 299
  # input_width = 299
  input_height = 160
  input_width = 160
  input_mean = 0
  input_std = 255
  input_layer = "Placeholder"
  output_layer = "final_result"

  graph = load_graph(model_file)

  t = read_tensor_from_image_file(
    file_name,
    input_height=input_height,
    input_width=input_width,
    input_mean=input_mean,
    input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

  with tf.Session(graph=graph) as sess:
    results = sess.run(output_operation.outputs[0], {
      input_operation.outputs[0]: t
    })

  results = np.squeeze(results)

  k = 1
  top_k = results.argsort()[-5:][::-1][:k]
  labels = load_labels(label_file)
  # for i in top_k:
  #   print(labels[i], results[i])
  return labels[top_k[0]], results[top_k[0]]


def test_images_classifier(path=os.path.join(st.ROOT,'res_test_data')):
  true_pos = 0
  total_num = 0
  total_t = 0
  true_label = path.split("/")[-2]
  if os.path.isfile(path):
    global count
    count += 1
    print("progress: " + str(count) + "/757")
    total_num += 1
    st_time = time.time()
    label, conf = single_imgae_classifier(path)
    t = time.time() - st_time
    print(t)
    total_t += t
    if label == true_label:
      true_pos += 1
    else:
      idx = wrong_classification_num[true_label]
      test_wrong_dir = os.path.join(st.ROOT,'test_wrong_data_mobilenet')
      test_wrong_name = str(true_label) + "-" + label + str(idx) + ".jpg"
      test_wrong_file = os.path.join(test_wrong_dir,test_wrong_name)
      idx += 1
      wrong_classification_num[true_label] = idx
      # print(path)
      # print(test_wrong_file)
      shutil.copy(path,test_wrong_file)
      print("---" + str(true_label) + "-->" + label + " confidence : {:.2f}".format(conf))
  else:
    files = os.listdir(path)
    for file in files:
      if not os.path.isfile(file):
        true_pos_child, total_num_child, total_t_child = test_images_classifier(os.path.join(path,file))
        true_pos += true_pos_child
        total_num += total_num_child
        total_t += total_t_child
  return true_pos, total_num, total_t



# def parse_args():
#   parser = argparse.ArgumentParser()
#   parser.add_argument("--image", help="image to be processed")
#   parser.add_argument("--graph", help="graph/model to be executed")
#   parser.add_argument("--labels", help="name of file containing labels")
#   parser.add_argument("--input_height", type=int, help="input height")
#   parser.add_argument("--input_width", type=int, help="input width")
#   parser.add_argument("--input_mean", type=int, help="input mean")
#   parser.add_argument("--input_std", type=int, help="input std")
#   parser.add_argument("--input_layer", help="name of input layer")
#   parser.add_argument("--output_layer", help="name of output layer")
#   args = parser.parse_args()
#   return args

if __name__ == "__main__":
  path2 = os.path.join(st.ROOT, 'res_test_data')
  path3 = os.path.join(path2,'cooks cottage')
  true_pos, total_num, total_t = test_images_classifier(path3)
  precision = true_pos/total_num
  time = total_t/total_num
  print("The precision is : "+str(precision))
  print("The average time is : "+str(time))
  print(wrong_classification_num)



