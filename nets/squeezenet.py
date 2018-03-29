# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim
import functools

import tensorflow as tf
import numpy as np

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

@slim.add_arg_scope
def _fire_module(inputs, 
                 squeeze_depth,
                 outputs_collections = None,
                 scope= None,
                 use_bypass=False 
                 ):
    """
    Creates a fire module
    
    Arguments:
        x                 : input
        nb_squeeze_filter : number of filters of squeeze. The filtersize of expand is 4 times of squeeze
        use_bypass        : if True then a bypass will be added
        name              : name of module e.g. fire123
    
    Returns:
        x                 : returns a fire module
    """
    
    with tf.variable_scope(scope, 'fire', [inputs]) as sc:
        with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
            expand_depth = squeeze_depth*4
            # squeeze
            squeeze = slim.conv2d(inputs, squeeze_depth, [1, 1], scope="squeeze_1X1")
            
            # expand
            expand_1x1 = slim.conv2d(squeeze, expand_depth, [1, 1], scope="expand_1x1")
            expand_3x3 = slim.conv2d(squeeze, expand_depth, [3, 3], scope="expand_3x3")
            
            # concat
            x_ret= tf.concat([expand_1x1, expand_3x3], axis=3)
            
            if use_bypass:
                x_ret = x_ret + inputs
        return slim.utils.collect_named_outputs(outputs_collections, sc.name, x_ret)

def squeezenet(inputs, 
               num_classes=1000, 
               compression=1.0,
               use_bypass=False,
               dropout_keep_prob=0.9,
               is_training=True,
               prediction_fn=tf.contrib.layers.softmax,
               spatial_squeeze=True,
               scope='SqueezeNet',
               global_pool=True):
    """
    Creating a SqueezeNet of version 1.0
    
    Arguments:
        input_shape  : shape of the input images e.g. (224,224,3)
        nb_classes   : number of classes
        use_bypass   : if true, bypass connections will be created at fire module 3, 5, 7, and 9 (default: False)
        dropout_rate : defines the dropout rate that is accomplished after last fire module (default: None)
        compression  : reduce the number of feature-maps (default: 1.0)
        
    Returns:
        Model        : Keras model instance
    """
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                     len(input_shape))
    
    end_points = {}
    with tf.variable_scope(scope, 'SqueezeNet', [inputs]) as sc: 
        end_points_collection = sc.original_name_scope + '_end_points' 
        with slim.arg_scope([slim.conv2d,slim.avg_pool2d,slim.max_pool2d,_fire_module],
                            outputs_collections = [end_points_collection]):
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                # conv1
                net = slim.conv2d(inputs, 96, [7, 7], stride=2, padding="SAME", scope="conv1")
               
                # maxpool1
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool1')
            
                # fire2
                net = _fire_module(net, int(16*compression), scope="fire2")
                
                # fire3
                net = _fire_module(net, int(16*compression), scope="fire3", use_bypass=use_bypass)
                
                # fire4
                net = _fire_module(net, int(32*compression), scope="fire4")
                
                # maxpool4
                net = slim.max_pool2d(net, [3, 3], stride=2, scope="maxpool4")
                
                # fire5
                net = _fire_module(net, int(32*compression), scope="fire5", use_bypass=use_bypass)
                
                # fire6
                net = _fire_module(net, int(48*compression), scope="fire6")
               
                # fire7
                net = _fire_module(net, int(48*compression), scope="fire7", use_bypass=use_bypass)
               
                # fire8
                net = _fire_module(net, int(64*compression), scope="fire8")
                
                # maxpool8
                net = slim.max_pool2d(net, [3, 3], stride=2, scope="maxpool8")
                
                # fire9
                net = _fire_module(net, int(64*compression), scope="fire9", use_bypass=use_bypass)

                # dropout
                if dropout_keep_prob:
                    net = slim.dropout(net,keep_prob=dropout_keep_prob, scope="dropout")

                # conv10
                net = slim.conv2d(net, num_classes, [1, 1], stride=1, padding="SAME", scope="conv10")
                
                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                
                # avgpool10
                if global_pool:
                    # Global average pooling.
                    net = slim.avg_pool2d(net, [13, 13], stride=1, scope="avgpool10")
                    end_points['global_pool'] = net
                if not num_classes:
                    return net, end_points
                
                # squeeze the axis
                if spatial_squeeze:
                    logits = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                    end_points["logits"]= logits
                
                if prediction_fn:
                    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
            
            return logits, end_points   
    
squeezenet.default_image_size = 224 

def squeezenet_arg_scope(is_training = True,
                    weight_decay = 0.0001,
                    batch_norm_decay = 0.997,
                    batch_norm_epsilon = 1e-5,
                    batch_norm_scale = True,
                    use_batch_norm=False):

    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }
    with  slim.arg_scope([slim.conv2d],
                   activation_fn=tf.nn.relu,
                   weights_regularizer=slim.l2_regularizer(weight_decay),
                   biases_initializer=tf.zeros_initializer(),
                   normalizer_fn=slim.batch_norm if use_batch_norm else None,
                   normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

# if __name__ == "__main__":
#     inputs = tf.random_normal([1, 224, 224, 3])
#     with slim.arg_scope(squeezenet_arg_scope()):
#          logits, end_points= squeezenet(inputs,1000,compression=1.0, use_bypass=True)
#      
#     writer = tf.summary.FileWriter("./logs_squeezenet", graph=tf.get_default_graph())
#     print("Layers")
#     for k, v in end_points.items():
#         print('name = {}, shape = {}'.format(v.name, v.get_shape()))
#      
#     print("Parameters")
#     for v in slim.get_model_variables():
#         print('name = {}, shape = {}'.format(v.name, v.get_shape()))
#          
#     init = tf.global_variables_initializer()
#     with tf.Session() as sess:
#         sess.run(init)
#         pred = sess.run(end_points['Predictions'])
# #         print(pred)
#         print(np.argmax(pred,1))
# #         print(pred[:,np.argmax(pred,1)])
