#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os

#%matplotlib inline
from matplotlib import pyplot as plt

import numpy as np
import os
import tensorflow as tf
import urllib2

from datasets import imagenet
from nets import resnet_v1
from preprocessing import vgg_preprocessing
import cv2


checkpoints_dir = '../tmp/checkpoints/'

slim = tf.contrib.slim


url = "http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz"
checkpoints_dir = '../tmp/checkpoints'

# if not tf.gfile.Exists(checkpoints_dir):
#     tf.gfile.MakeDirs(checkpoints_dir)
#  
# dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

# 网络模型的输入图像有默认的尺寸
# 因此，我们需要先调整输入图片的尺寸
image_size = resnet_v1.resnet_v1_50.default_image_size

with tf.Graph().as_default():
    
    image =cv2.imread("First_Student_IC_school_bus_202076.jpg")
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)# change channel
#     image = image [:, :, (2, 1, 0)] # change channel
    
    
    # 对图片做缩放操作，保持长宽比例不变，裁剪得到图片中央的区域
    # 裁剪后的图片大小等于网络模型的默认尺寸
    processed_image = vgg_preprocessing.preprocess_image(image,
                                                         image_size,
                                                         image_size,
                                                         is_training=False)
    
    # 可以批量导入图像
    # 第一个维度指定每批图片的张数
    # 我们每次只导入一张图片
    processed_images  = tf.expand_dims(processed_image, 0)
    
    # 创建模型，使用默认的arg scope参数
    # arg_scope是slim library的一个常用参数
    # 可以设置它指定网络层的参数，比如stride, padding 等等。
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits, _ = resnet_v1.resnet_v1_50(processed_images,
                               num_classes=1000,
                               is_training=False)
    
    # 我们在输出层使用softmax函数，使输出项是概率值
    probabilities = tf.nn.softmax(logits)
    
    # 创建一个函数，从checkpoint读入网络权值
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'resnet_v1_50.ckpt'),
        slim.get_model_variables('resnet_v1_50'))
    
    with tf.Session() as sess:
        # 加载权值
        init_fn(sess)
        
        ops = sess.graph.get_operations()
        for op in ops:
            print(op.name)
            
        print("Parameters")
        for v in slim.get_model_variables():
            print('name = {}, shape = {}'.format(v.name, v.get_shape()))
        

        writer = tf.summary.FileWriter("./logs_resnet", graph=tf.get_default_graph())
        
        print("Finish!")
     
        # 图片经过缩放和裁剪，最终以numpy矩阵的格式传入网络模型
        network_input, probabilities = sess.run([processed_image,
                                                probabilities])
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
                                            key=lambda x:x[1])]
     
     
     
    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
        # 打印top5的预测类别和相应的概率值。
        print('Probability %0.2f => [%s]' % (probabilities[index], names[index+1]))


