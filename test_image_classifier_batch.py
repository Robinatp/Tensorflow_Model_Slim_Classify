from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
import math
import time
import numpy as np
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', 'tmp/inception_finetuned/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')
tf.app.flags.DEFINE_string(
    'test_list', '', 'Test image list.')
tf.app.flags.DEFINE_string(
    'test_dir', '.', 'Test image directory.')
tf.app.flags.DEFINE_integer(
    'batch_size', 16, 'Batch size.')
tf.app.flags.DEFINE_integer(
    'num_classes', 5, 'Number of classes.')
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
    'test_image_size', None, 'Eval image size')
FLAGS = tf.app.flags.FLAGS
def main(_):
    if not FLAGS.test_list:
        raise ValueError('You must supply the test list with --test_list')
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()
        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(FLAGS.num_classes - FLAGS.labels_offset),
            is_training=False)
        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)
        test_image_size = FLAGS.test_image_size or network_fn.default_image_size
        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path
        batch_size = FLAGS.batch_size
        tensor_input = tf.placeholder(tf.float32, [None, test_image_size, test_image_size, 3])
        logits, _ = network_fn(tensor_input)
        logits = tf.nn.top_k(logits, 5)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        test_ids = [line.strip() for line in open(FLAGS.test_list)]
        tot = len(test_ids)
        results = list()
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
            time_start = time.time()
            
            for idx in range(0, tot, batch_size):
                images = list()
                idx_end = min(tot, idx + batch_size)
                print(idx)
                for i in range(idx, idx_end):
                    image_id = test_ids[i]
                    test_path = os.path.join(FLAGS.test_dir, image_id)
                    image = open(test_path, 'rb').read()
                    image = tf.image.decode_jpeg(image, channels=3)
                    processed_image = image_preprocessing_fn(image, test_image_size, test_image_size)
                    processed_image = sess.run(processed_image)
                    images.append(processed_image)
                images = np.array(images)
                predictions = sess.run(logits, feed_dict = {tensor_input : images}).indices
                for i in range(idx, idx_end):
                    print('{} {}'.format(image_id, predictions[i - idx].tolist()))
                
            time_end = time.time()
            time_total = time_end - time_start
            
            print('total time: {}, total images: {}, average time: {}'.format(
                time_total, len(test_ids), time_total / len(test_ids)))
                          
if __name__ == '__main__':
    tf.app.run()