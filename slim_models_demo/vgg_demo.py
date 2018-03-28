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
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

checkpoints_dir = '../tmp/checkpoints/'

slim = tf.contrib.slim

#download the vgg_16_2016_08_28.tar.gz checkpoint from models
url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"
checkpoints_dir = '../tmp/checkpoints'

# if not tf.gfile.Exists(checkpoints_dir):
#     tf.gfile.MakeDirs(checkpoints_dir)
#  
# dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)


#set the default image_seze
image_size = vgg.vgg_16.default_image_size



def save_graph_to_file(sess, graph, graph_file_name):
  output_graph_def = graph_util.convert_variables_to_constants(
      sess, graph, ["Softmax"])
  with gfile.FastGFile(graph_file_name, 'wb') as f:
    f.write(output_graph_def.SerializeToString())
  return

with tf.Graph().as_default():
    
    url = ("https://upload.wikimedia.org/wikipedia/commons/d/d9/"
           "First_Student_IC_school_bus_202076.jpg")
    
    # connect the internet and download it
    image_string = urllib2.urlopen(url).read()
    
    #decode the image
    image = tf.image.decode_jpeg(image_string, channels=3)
    
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
        
        print("network operation")
        ops = sess.graph.get_operations()
        for op in ops:
            print(op.name)
        save_graph_to_file(sess,sess.graph_def ,os.path.join(checkpoints_dir, 'vgg_16_freeze_graph.pb'))  
        # 图片经过缩放和裁剪，最终以numpy矩阵的格式传入网络模型
        np_image, network_input, probabilities = sess.run([image,
                                                           processed_image,
                                                           probabilities])
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
                                            key=lambda x:x[1])]
    
    # 显示下载的图片
    plt.figure()
    plt.imshow(np_image.astype(np.uint8))
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
    
    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
        # 打印top5的预测类别和相应的概率值。
        print('Probability %0.2f => [%s]' % (probabilities[index], names[index+1]))

    res = slim.get_model_variables()

