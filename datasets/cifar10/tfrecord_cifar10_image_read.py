
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from cifar10 import Cifar10DataSet

import os
import sys

# This is needed since the notebook is stored in the object_detection folder.
TF_API="/home/robin/eclipse-workspace-python/TF_models/models/research/slim"
sys.path.append(os.path.split(TF_API)[0])
sys.path.append(TF_API)

from datasets import cifar10
from datasets import flowers
from datasets import imagenet
from datasets import mnist
from datasets import mydata
from preprocessing import cifarnet_preprocessing
from preprocessing import inception_preprocessing
from preprocessing import lenet_preprocessing
from preprocessing import vgg_preprocessing


from tensorflow.contrib import slim

cifar10_data_dir = "/home/robin/Dataset/cifar10"


if __name__ == "__main__":
    with tf.Graph().as_default(): 
        subset = 'train' 
        dataset = Cifar10DataSet(cifar10_data_dir, subset, False)
        image_batch, label_batch = dataset.make_batch(2)
        print(image_batch, label_batch)


        
                    
        with tf.Session() as sess:    
            with slim.queues.QueueRunners(sess):
                for i in range(10):
                    np_image, np_labels = sess.run([image_batch, label_batch])

                                                                                                                             
#                     plt.figure()
#                     plt.imshow(np_image[0])
#                     plt.title('%s, %d x %d' % (name, height, width))
#                     plt.axis('off')
#                     plt.show()
 
                    print("labels :", dataset.labels_to_names[np_labels[0]])
                    print("--------------")
                    cv2.imshow('image:',np_image[0]/255 )
    
                    cv2.waitKey(0)
                     
                   
                    
                    
                    
                    
