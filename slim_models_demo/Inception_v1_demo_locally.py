
import numpy as np
import os
import tensorflow as tf


#%matplotlib inline
from matplotlib import pyplot as plt
try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib
import cv2
from datasets import imagenet
from nets import inception
from preprocessing import inception_preprocessing

from tensorflow.contrib import slim

from datasets import dataset_utils

url = "http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz"
checkpoints_dir = '../tmp/checkpoints'

# if not tf.gfile.Exists(checkpoints_dir):
#     tf.gfile.MakeDirs(checkpoints_dir)
#  
# dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

image_size = inception.inception_v1.default_image_size

with tf.Graph().as_default():
    image =cv2.imread("First_Student_IC_school_bus_202076.jpg")
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)# change channel
    image = tf.cast(image, tf.float32)
#     print(image.dtype)
#     image = tf.image.decode_jpeg(tf.read_file("First_Student_IC_school_bus_202076.jpg"), channels=3)
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    processed_images  = tf.expand_dims(processed_image, 0)
    
    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits, _ = inception.inception_v1(processed_images, num_classes=1001, is_training=False)
    probabilities = tf.nn.softmax(logits)
    
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'inception_v1.ckpt'),
        slim.get_model_variables('InceptionV1'))
    
    saver = tf.train.Saver()#method 2 for restore
    
    with tf.Session() as sess:
#         init_fn(sess)#method 1 for restore
        saver.restore(sess, os.path.join(checkpoints_dir, 'inception_v1.ckpt'))#method 2 for restore
        
        np_image ,network_input ,probabilities = sess.run([image,processed_image,probabilities])
        
        print(probabilities.shape)
        probabilities = probabilities[0,:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
                                        key=lambda x:x[1])]    
        
   
    plt.figure()
    plt.imshow(np_image.astype(np.uint8))
    plt.suptitle("Downloaded image", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()

    # to show the image.
    plt.imshow( network_input)
    plt.suptitle("Resized, Cropped and Mean-Centered input to network",
                 fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()

    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
        print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index]))