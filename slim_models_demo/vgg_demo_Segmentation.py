#coding=utf-8
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

from preprocessing import vgg_preprocessing


checkpoints_dir = '../tmp/checkpoints/'

slim = tf.contrib.slim
# 加载像素均值及相关函数
from preprocessing.vgg_preprocessing import (_mean_image_subtraction,
                                            _R_MEAN, _G_MEAN, _B_MEAN)

# 展现分割结果的函数，以不同的颜色区分各个类别
def discrete_matshow(data, labels_names=[], title=""):
    #获取离散化的色彩表
    cmap = plt.get_cmap('Paired', np.max(data)-np.min(data)+1)
    mat = plt.matshow(data,
                      cmap=cmap,
                      vmin = np.min(data)-.5,
                      vmax = np.max(data)+.5)
    #在色彩表的整数刻度做记号
    cax = plt.colorbar(mat,
                       ticks=np.arange(np.min(data),np.max(data)+1))

    # 添加类别的名称
    if labels_names:
        cax.ax.set_yticklabels(labels_names)

    if title:
        plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.show()

with tf.Graph().as_default():

    url = ("https://upload.wikimedia.org/wikipedia/commons/d/d9/"
           "First_Student_IC_school_bus_202076.jpg")

    image_string = urllib2.urlopen(url).read()
    image = tf.image.decode_jpeg(image_string, channels=3)

    # 减去均值之前，将像素值转为32位浮点
    image_float = tf.to_float(image, name='ToFloat')

    # 每个像素减去像素的均值
    processed_image = _mean_image_subtraction(image_float,
                                              [_R_MEAN, _G_MEAN, _B_MEAN])

    input_image = tf.expand_dims(processed_image, 0)

    with slim.arg_scope(vgg.vgg_arg_scope()):

        # spatial_squeeze选项指定是否启用全卷积模式
        logits, _ = vgg.vgg_16(input_image,
                               num_classes=1000,
                               is_training=False,
                               spatial_squeeze=False)

    # 得到每个像素点在所有1000个类别下的概率值，挑选出每个像素概率最大的类别
    # 严格说来，这并不是概率值，因为我们没有调用softmax函数
    # 但效果等同于softmax输出值最大的类别
    pred = tf.argmax(logits, dimension=3)

    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
        slim.get_model_variables('vgg_16'))

    with tf.Session() as sess:
        init_fn(sess)
        segmentation, np_image = sess.run([pred, image])

    # 去除空的维度
    segmentation = np.squeeze(segmentation)
    
    unique_classes, relabeled_image = np.unique(segmentation,
                                                return_inverse=True)
    
    segmentation_size = segmentation.shape
    
    relabeled_image = relabeled_image.reshape(segmentation_size)
    
    labels_names = []
    names = imagenet.create_readable_names_for_imagenet_labels()

    
    for index, current_class_number in enumerate(unique_classes):
    
        labels_names.append(str(index) + ' ' + names[current_class_number+1])
    
    discrete_matshow(data=relabeled_image, labels_names=labels_names, title="Segmentation")
    
    res = slim.get_model_variables()
    