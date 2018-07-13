"""MobileNet v2.

MobileNet is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and different
head (for example: embeddings, localization and classification).

This is a revised version based on (https://arxiv.org/pdf/1801.04381.pdf)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools

import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np
import time



def mobilenet_v2_arg_scope(weight_decay=0.00004, is_training = True, stddev = 0.15, regularize_depthwise=True, dropout_prob=0.999):
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    weights_init = tf.truncated_normal_initializer(stddev=stddev)
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None
   
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_initializer=weights_init,
                        activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'center': True, 'scale': True, 'decay': 0.997, 'epsilon': 0.001}):
        with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
            with slim.arg_scope([slim.separable_conv2d],
                                weights_regularizer=depthwise_regularizer, depth_multiplier=1.0):
                with slim.arg_scope([slim.dropout], is_training=is_training, keep_prob=dropout_prob) as sc:
                    return sc


# def mobilenet_v2_arg_scope(weight_decay=0.00004,
#                    is_training=True,
#                    stddev=0.09,
#                    dropout_keep_prob=0.8,
#                    bn_decay=0.997):
#   """Defines Mobilenet training scope.
#   Args:
#     is_training: if set to False this will ensure that all customizations are
#     set to non-training mode. This might be helpful for code that is reused
#     across both training/evaluation, but most of the time training_scope with
#     value False is not needed.
#  
#     weight_decay: The weight decay to use for regularizing the model.
#     stddev: Standard deviation for initialization, if negative uses xavier.
#     dropout_keep_prob: dropout keep probability
#     bn_decay: decay for the batch norm moving averages.
#  
#   Returns:
#     An argument scope to use via arg_scope.
#   """
#   # Note: do not introduce parameters that would change the inference
#   # model here (for example whether to use bias), modify conv_def instead.
#   batch_norm_params = {
#       'center': True, 
#       'scale': True,
#       'is_training': is_training,
#       'decay': bn_decay,
#   }
#  
#   if stddev < 0:
#     weight_intitializer = slim.initializers.xavier_initializer()
#   else:
#     weight_intitializer = tf.truncated_normal_initializer(stddev=stddev)
#  
#   # Set weight_decay for weights in Conv and FC layers.
#   with slim.arg_scope(
#       [slim.conv2d, slim.fully_connected, slim.separable_conv2d],
#       weights_initializer=weight_intitializer,
#       normalizer_fn=slim.batch_norm), \
#       slim.arg_scope([slim.batch_norm], **batch_norm_params), \
#       slim.arg_scope([slim.dropout], is_training=is_training,
#                      keep_prob=dropout_keep_prob), \
#       slim.arg_scope([slim.conv2d], \
#                      weights_regularizer=slim.l2_regularizer(weight_decay)), \
#       slim.arg_scope([slim.separable_conv2d], weights_regularizer=None) as sc:
#       return sc


# inverted residual with linear bottleneck
@slim.add_arg_scope
def inverted_residual_bottleneck(input, input_depth, depth, expansion, stride, outputs_collections = None, scope= None):
    '''
    Bottleneck residual block transforming from k to k' channels, with stride s, and expansion factor t.
    '''
    with tf.variable_scope(scope, 'inverted_residual', [input]) as sc:
        # fundamental network struture of inverted_residual_block
        shutcut = input
        if expansion == 1:#first bottleneck only include depth-separable convolution,
            # depthwise conv2d
            net = slim.separable_conv2d(input, num_outputs=None, kernel_size=[3, 3], stride=stride, scope="depthwise")
            # pointwise conv2d, project feature back to k
            net = slim.conv2d(net, num_outputs=depth, kernel_size=[1, 1], activation_fn=None, scope="project")
            out = net
            return slim.utils.collect_named_outputs(outputs_collections, sc.name, out)
            
        # pointwise conv2d, expand feature up to 6 times ( recorded in mobilenetv2 paper )
        net = slim.conv2d(input, num_outputs=input_depth * expansion, kernel_size=[1, 1], scope="expand")
        # depthwise conv2d
        net = slim.separable_conv2d(net, num_outputs=None, kernel_size=[3, 3], stride=stride, scope="depthwise")
        # pointwise conv2d, project feature back from 6 times to k'
        net = slim.conv2d(net, num_outputs=depth, kernel_size=[1, 1], activation_fn=None, scope="project")
        # stride 2 blocks, there is not a residual block
        if stride == 2:
            out = net
            return slim.utils.collect_named_outputs(outputs_collections, sc.name, out)
        # stride 1 block with a residual block that the input acts a shutcut
        else:
            if input_depth != depth:#the case of Dimension from 64 up 96 without stride=2,only the dimension groups up
                out = net
                return slim.utils.collect_named_outputs(outputs_collections, sc.name, out)
#                 shutcut = slim.conv2d(shutcut, depth, kernel_size=[1, 1], activation_fn=None)
#                 return tf.add(net, shutcut)
            out = tf.add(net, shutcut)
            return slim.utils.collect_named_outputs(outputs_collections, sc.name, out)


# repeated inverted residual bottleneck block
def bottleneck_stages(inputs, expand, depth, repeat, stride, index, scope= None):
    with tf.variable_scope(scope,"Stage",[inputs]) as sc:
        input_depth = inputs.get_shape().as_list()[-1]
        # first layer needs to consider stride,except for inverted_residual_block0 and inverted_residual_block10_12, all others use stride 2
        net = inverted_residual_bottleneck(inputs, input_depth, depth, expand, stride,scope='bottleneck_{}'.format(0+index))
    
        for i in range(1, repeat):
            net = inverted_residual_bottleneck(net, depth, depth, expand, 1,scope='bottleneck_{}'.format(i+index))
    
        return net


def mobilenet_v2_base(inputs,
                      expand = 6,
                      min_depth=8,
                      depth_multiplier=1.0,
                      scope=None):
    endpoints = dict()
    multiplier_depth = lambda d: max(int(d * depth_multiplier), min_depth)
    neural_net = slim.conv2d(inputs, num_outputs=multiplier_depth(32), kernel_size=[3, 3], stride=2)
    #inverted_residual_block0
    neural_net = bottleneck_stages(neural_net, expand=     1, depth=multiplier_depth(16), repeat=1, stride=1, index=0, scope='Stage{}'.format(0)) 
    #inverted_residual_block1_2
    neural_net = bottleneck_stages(neural_net, expand=expand, depth=multiplier_depth(24), repeat=2, stride=2, index=1, scope='Stage{}'.format(1)) 
    #inverted_residual_block3_5
    neural_net = bottleneck_stages(neural_net, expand=expand, depth=multiplier_depth(32), repeat=3, stride=2, index=3, scope='Stage{}'.format(2))  
    #inverted_residual_block6_9
    neural_net = bottleneck_stages(neural_net, expand=expand, depth=multiplier_depth(64), repeat=4, stride=2, index=6, scope='Stage{}'.format(3))
    #inverted_residual_block10_12
    neural_net = bottleneck_stages(neural_net, expand=expand, depth=multiplier_depth(96), repeat=3, stride=1, index=10, scope='Stage{}'.format(4))
    #inverted_residual_block13_15
    neural_net = bottleneck_stages(neural_net, expand=expand, depth=multiplier_depth(160),repeat=3, stride=2, index=13, scope='Stage{}'.format(5))
    #inverted_residual_block16
    neural_net = bottleneck_stages(neural_net, expand=expand, depth=multiplier_depth(320),repeat=1, stride=1, index=16, scope='Stage{}'.format(6))
    neural_net = slim.conv2d(neural_net, num_outputs=multiplier_depth(1280), kernel_size=[1, 1])
    endpoints['bottleneck'] = neural_net

    return neural_net, endpoints


def mobilenet_v2(inputs,
                 num_classes=1001,
                 dropout_keep_prob=0.90,
                 is_training=True,
                 min_depth=8,
                 depth_multiplier=1.0,
                 prediction_fn=tf.contrib.layers.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='MobilenetV2',
                 global_pool=False):
  input_shape = inputs.get_shape().as_list()
  if len(input_shape) != 4:
      raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                     len(input_shape))
  end_points = {}
  expand = 6
  with tf.variable_scope(scope, 'MobilenetV2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points' 
        with slim.arg_scope([slim.conv2d,slim.avg_pool2d,slim.max_pool2d,inverted_residual_bottleneck],outputs_collections = [end_points_collection]):
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.separable_conv2d] ,activation_fn=tf.nn.relu6):
                with slim.arg_scope([slim.separable_conv2d],weights_regularizer=None, depth_multiplier=1.0):
                    with slim.arg_scope([slim.dropout,slim.batch_norm], is_training=is_training):
                        net, end_points = mobilenet_v2_base(inputs, 
                                              min_depth=min_depth,
                                              depth_multiplier=depth_multiplier,
                                              scope=sc)
                        
                        # Convert end_points_collection into a dictionary of end_points.
                        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                        
                        net = tf.identity(net, name='embedding')
                        
                        with tf.variable_scope('Logits'):
                            if global_pool:
                                # Global average pooling.
                                net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                                end_points['global_pool'] = net
                            else:
                                # Pooling with a fixed kernel size.
                                kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
                                net = slim.avg_pool2d(net, kernel_size, padding='VALID',scope='AvgPool_1a')
                                end_points['AvgPool_1a'] = net
                            if not num_classes:
                                return net, end_points

                            # 1 x 1 x k
                            net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout')
                            logits = slim.conv2d(net, num_classes, [1, 1], 
                                                 activation_fn=None, 
                                                 normalizer_fn=None,
                                                 scope='features')
                            
                            if spatial_squeeze:
                                logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
                            logits = tf.identity(logits, name='output')
                        end_points['Logits'] = logits
                        if prediction_fn:
                            end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points

mobilenet_v2.default_image_size = 224


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


mobilenet_v2_075 = wrapped_partial(mobilenet_v2, depth_multiplier=0.75)
mobilenet_v2_050 = wrapped_partial(mobilenet_v2, depth_multiplier=0.50)
mobilenet_v2_025 = wrapped_partial(mobilenet_v2, depth_multiplier=0.25)


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.
  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1])]
  return kernel_size_out


  
if __name__ == "__main__":
    inputs = tf.random_normal([1, 224, 224, 3])
    
    with slim.arg_scope(mobilenet_v2_arg_scope(is_training=False)):
        logits, end_points= mobilenet_v2_025(inputs,100)   
      
    writer = tf.summary.FileWriter("./logs", graph=tf.get_default_graph())
    print("Layers")
    for k, v in end_points.items():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))
      
    print("Parameters")
    for v in slim.get_model_variables():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))
          
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
#         print(sess.run(logits))
        pred = sess.run(end_points['Predictions'])
        print(pred)
        print(np.argmax(pred,1))
        print(pred[:,np.argmax(pred,1)])
        
        
        cnt = 0
        for i in range(101):
            t1 = time.time()
            output = sess.run(end_points['Predictions'])
            if i != 0:
                cnt += time.time() - t1
        print(cnt / 100)
