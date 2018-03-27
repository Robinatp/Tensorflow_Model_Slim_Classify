from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import tensorflow as tf
import numpy as np
from nets import nets_factory
from preprocessing import preprocessing_factory
from datasets import imagenet
slim = tf.contrib.slim
'''
usage for test_image_classifier.py

python test_image_classifier.py \
--checkpoint_path={your checkpoint path or ckpt file} \
--test_path={your test path} \
--num_classes={your class classifier} \
--model_name={your model name}
'''
tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '../tmp/checkpoints/with_placeholder',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'test_path', 'First_Student_IC_school_bus_202076.jpg', 'Test image path.')

tf.app.flags.DEFINE_integer(
    'num_classes', 1000, 'Number of classes.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'vgg_16', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'test_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


def main(_):
    if not FLAGS.test_path:
        raise ValueError('You must supply the test list with --test_path')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
#         tf_global_step = slim.get_or_create_global_step()

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

	###########################
        # get the checkpoint file #
        ###########################
        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path
	print("restore from",checkpoint_path) 
	
        tf.Graph().as_default()
        with tf.Session() as sess:
	    ################################
            # open the file and preprocess #
            ################################
            image = open(FLAGS.test_path, 'rb').read()
            image = tf.image.decode_jpeg(image, channels=3)
            processed_image = image_preprocessing_fn(image, test_image_s ize, test_image_size)
            processed_images = tf.expand_dims(processed_image, 0)
            
	    #############################################
            # build the network and restore the network #
            #############################################
            logits, _ = network_fn(processed_images)
            probabilities = tf.nn.softmax(logits)
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
            
            np_image, network_input, predictions = sess.run([image, processed_image, probabilities])
            probabilities = np.squeeze(predictions,0)
            names = imagenet.create_readable_names_for_imagenet_labels()
            
            pre = np.argmax(probabilities, axis=0)
            print('{} {}  {}'.format(FLAGS.test_path,pre ,names[pre+1]))
            top_k = probabilities.argsort()[-5:][::-1]
            for index in top_k:
                print('Probability %0.2f => [%s]' % (probabilities[index], names[index+1]))

if __name__ == '__main__':
    tf.app.run()
