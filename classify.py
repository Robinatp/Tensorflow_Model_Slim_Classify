#@title Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import csv
import time
from datetime import datetime
import tensorflow as tf
import re


class ClassifyModel(object):
  """Class to load classify model and run inference."""
  def __init__(self, frozen_graph=None, input_tensor_name=None, output_tensor_name=None):
    """Creates and loads pretrained  model."""
    if not frozen_graph.endswith('.pb'):
        raise ValueError('frozen_graph is not a correct pb file!')
    if ((not input_tensor_name) or (not output_tensor_name)) :
        raise ValueError('input_tensor_name or output_tensor_name is None!')
    
    self.input_tensor_name = input_tensor_name
    self.output_tensor_name = output_tensor_name
    self.is_debug = True
    self.prediction =-1
    
    self.graph, \
    self.input_tensor, \
    self.output_tensor = self.load_graph(frozen_graph)
    self.sess = tf.Session(graph=self.graph)
    self.labels = self.load_labels("data/imagenet_slim_labels.txt")
    
  def load_graph(self, frozen_graph):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        input_tensor, output_tensor = tf.import_graph_def(
                                                graph_def, 
                                                input_map=None, 
                                                return_elements=[self.input_tensor_name, self.output_tensor_name], 
                                                name="",
                                                producer_op_list=None)   
        if self.is_debug:
            writer = tf.summary.FileWriter("./logs_classify_graph", graph=graph)
            writer.close() 
        
    return graph, input_tensor, output_tensor

  def read_tensor_from_image_file(self,
                                file_name,
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
      with tf.Session() as sess:
          result = sess.run(normalized)
    
      return result


  def load_labels(self, label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


  def run(self, image):
    """Runs inference on a single image.
    Args:
      image:  raw input image.
    Returns:
    """
    start_time = time.time()
    results= self.sess.run(
        self.output_tensor,
        feed_dict={self.input_tensor:image})
    duration = time.time() - start_time
    if  self.is_debug:
        print ('%s: ClassifyModel.run(), duration = %.3f' %(datetime.now(), duration))
    self.prediction = np.argmax(results)
    return results


def vis_segmentation(image):
  plt.figure(figsize=(6, 6))
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')
  plt.show()


def run_classification(ClassifyModel,path):
    """Inferences classify model and visualizes result."""
    if not tf.gfile.Exists(path):
        raise ValueError('path  is None!')
    
    image_np = ClassifyModel.read_tensor_from_image_file(path)
    results = ClassifyModel.run(image_np)
    labels = ClassifyModel.labels[ClassifyModel.prediction]
    if ClassifyModel.is_debug:
        results = np.squeeze(results)
        top_k = results.argsort()[-5:][::-1]
        for i in top_k:
            print(ClassifyModel.labels[i], results[i])
    #vis_segmentation(np.squeeze(image_np, axis=0))
    return labels


def read_examples_list(path):
  """Read list of training or validation examples.
  Args:
    path: absolute path to examples list file.

  Returns:
    list of example identifiers (strings).
  """
  with tf.gfile.GFile(path) as fid:
    lines = fid.readlines()
  return [line.strip().split(' ')[0] for line in lines]



MODEL_DIR= "data/inception_v3_2016_08_28_frozen.pb"

flags = tf.app.flags
# Dataset settings.

tf.app.flags.DEFINE_string('test_path', 'data/', 'Test image path.')

flags.DEFINE_string('modir_dir', MODEL_DIR, 'Where the Model reside.')

FLAGS = flags.FLAGS


def main():
    INPUT_TENSOR_NAME = 'input:0'
    OUTPUT_TENSOR_NAME = 'InceptionV3/Predictions/Reshape_1:0'

    MODEL = ClassifyModel(FLAGS.modir_dir, INPUT_TENSOR_NAME,OUTPUT_TENSOR_NAME)
    
    image_files = tf.gfile.Glob(FLAGS.test_path+"*.jpg")
    print(image_files)
    
    for file in image_files:
        run_classification(MODEL, file)

main()



