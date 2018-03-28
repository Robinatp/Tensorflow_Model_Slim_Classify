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
from nets import vgg
from preprocessing import vgg_preprocessing
import cv2

checkpoints_dir = '../tmp/checkpoints/'

slim = tf.contrib.slim



# 网络模型的输入图像有默认的尺寸
# 因此，我们需要先调整输入图片的尺寸
image_size = vgg.vgg_16.default_image_size

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
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, _ = vgg.vgg_16(processed_images,
                               num_classes=1000,
                               is_training=False)
    
    # 我们在输出层使用softmax函数，使输出项是概率值
    probabilities = tf.nn.softmax(logits)
    
    # 创建一个函数，从checkpoint读入网络权值
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
        slim.get_model_variables('vgg_16'))
    
    with tf.Session() as sess:
    
        # 加载权值
        init_fn(sess)
    
        # 图片经过缩放和裁剪，最终以numpy矩阵的格式传入网络模型
        network_input, probabilities = sess.run([processed_image,
                                                probabilities])
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
                                            key=lambda x:x[1])]
    
    # 显示下载的图片
    plt.figure()
    plt.imshow(image.astype(np.uint8))
    plt.suptitle("Downloaded image", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()
    
    
    # 显示最终传入网络模型的图片
    # 图像的像素值做了[-1, 1]的归一化
    # to show the image.
    plt.imshow( network_input / (network_input.max() - network_input.min()) )
    plt.suptitle("Resized, Cropped and Mean-Centered input to network",
                 fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()
    
    cv2.imshow("Downloaded image",image)
    cv2.imshow("Resized, Cropped and Mean-Centered input to network",network_input)
    cv2.waitKey(0)
    
    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
        # 打印top5的预测类别和相应的概率值。
        print('Probability %0.2f => [%s]' % (probabilities[index], names[index+1]))

#     res = slim.get_model_variables()

