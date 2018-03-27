# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim
import functools

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

def channel_shuffle(input, output, group, scope=None):
        assert 0 == output % group, "Output channels must be a multiple of groups"
        num_channels_in_group = output // group
        with tf.variable_scope(scope,"ChannelShuffle",[input]):
            net = tf.split(input, output, axis=3, name="split")
            chs = []
            for i in range(group):
                for j in range(num_channels_in_group):
                    chs.append(net[i + j * group])
            net = tf.concat(chs, axis=3, name="concat")
        return net
    
def channel_shuffle_v1(input,depth_bottleneck,group,scope=None):
    assert 0 == depth_bottleneck % group, "Output channels must be a multiple of groups"
    with tf.variable_scope(scope,"ChannelShuffle",[input]):
        n, h, w, c =input.shape.as_list()
        x_reshape = tf.reshape(input, [-1,h,w,group,depth_bottleneck//group])
        x_transposed =tf.transpose(x_reshape, [0,1,2,4,3])
        net = tf.reshape(x_transposed, [-1,h,w,c])
        return net
        
def group_pointwise_conv2d(inputs, depth, stride, group, relu=True, scope=None):
    
        assert 0 == depth % group, "Output channels must be a multiple of groups"
        num_channels_in_group = depth // group
        with tf.variable_scope(scope, 'GConv', [inputs]) as sc:
            net = tf.split(inputs, group, axis=3, name="split")
            for i in range(group):
                net[i] = slim.conv2d(net[i],
                                     num_channels_in_group,
                                     [1, 1],
                                     stride=stride,
                                     activation_fn=None,
                                     normalizer_fn=None)
            net = tf.concat(net, axis=3, name="concat")
            net = slim.batch_norm(net, activation_fn = tf.nn.relu if relu else None)    
        return net    
    
@slim.add_arg_scope
def shuffle_bottleneck(inputs, depth_bottleneck, group, stride, shuffle=True, outputs_collections = None, scope= None):
        if 1 != stride:
            _b, _h, _w, _c = inputs.get_shape().as_list()
            depth_bottleneck = depth_bottleneck - _c

        assert 0 == depth_bottleneck % group, "Output channels must be a multiple of groups"

        with tf.variable_scope(scope, 'Unit', [inputs]) as sc:
            print("shuffle_bottleneck",sc.name)
            if 1 != stride:
                net_skip = slim.avg_pool2d(inputs, [3, 3], stride, padding="SAME", scope='3x3AVGPool2D')
            else:
                net_skip = inputs

            net = group_pointwise_conv2d(inputs, depth_bottleneck, 1, group = (1 if (cmp(sc.name ,'ShuffleNet/Stage2/Unit0')== 0) else group), relu=True, scope="1x1GConvIn")

            if shuffle:
                net = channel_shuffle_v1(net, depth_bottleneck, group)

            with tf.variable_scope("3x3DWConv"):
                depthwise_filter = tf.get_variable("depth_conv_w", [3, 3, depth_bottleneck, 1],
                                                   initializer=tf.truncated_normal_initializer(stddev=0.01))
                net = tf.nn.depthwise_conv2d(net, depthwise_filter, [1, stride, stride, 1], 'SAME', name="DWConv")
                # Todo: Add batch norm here
                net = slim.batch_norm(net, activation_fn = None)

            net = group_pointwise_conv2d(net, depth_bottleneck, 1, group,relu=False, scope="1x1GConvOut")
            

            if 1 != stride:
                net = tf.concat([net, net_skip], axis=3)
            else:
                net = net + net_skip
            out = tf.nn.relu(net)
        return slim.utils.collect_named_outputs(outputs_collections, sc.name, out)
    
   
def shuffle_stage(inputs, depth, groups, repeat, shuffle=True, scope=None):
     with tf.variable_scope(scope,"Stage",[inputs]) as sc:
        net = shuffle_bottleneck(inputs, depth, group = groups, stride = 2, shuffle=shuffle , scope='Unit{}'.format(0))
        
        for i in range(repeat):
            net = shuffle_bottleneck(net, depth, group = groups, stride = 1, shuffle=shuffle , scope='Unit{}'.format(i + 1))   
        return net   

def shufflenet(inputs, 
               num_classes=None, 
               shuffle=True, 
               base_ch=144, 
               groups=1, 
               is_training=True,
               prediction_fn=tf.contrib.layers.softmax,
               scope=None):
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                     len(input_shape))
          
    with tf.variable_scope(scope, 'ShuffleNet', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        print(end_points_collection)    
        with slim.arg_scope([slim.conv2d,slim.avg_pool2d,slim.max_pool2d,shuffle_bottleneck],
                            outputs_collections = [end_points_collection]):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
            
                with tf.variable_scope('Stage1'):
                    net = slim.conv2d(inputs, 24, [3, 3], stride = 2, scope= "conv1")
                    net = slim.max_pool2d(net, [3, 3], stride = 2, padding='SAME', scope= "pool1")
        
                net = shuffle_stage(net, depth = base_ch * 1, groups = groups, repeat = 3, scope='Stage2')
                net = shuffle_stage(net, depth = base_ch * 2, groups = groups, repeat = 7, scope='Stage3')
                net = shuffle_stage(net, depth = base_ch * 4, groups = groups, repeat = 3, scope='Stage4')
    
                
                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        
                with tf.variable_scope('Stage5'):
                    net = tf.reduce_mean(net, [1, 2],name='global_pool')
                    end_points['global_pool'] = net
                
                    logits = slim.fully_connected(net, num_classes,scope='logits')
                    end_points[sc.name + '/logits'] = logits
                    if prediction_fn:
                        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')


                return logits, end_points

shufflenet.default_image_size = 224    


def wrapped_partial(func, *args, **kwargs):
  partial_func = functools.partial(func, *args, **kwargs)
  functools.update_wrapper(partial_func, func)
  return partial_func

shufflenet_g1 = wrapped_partial(shufflenet, base_ch=144, groups=1)
shufflenet_g2 = wrapped_partial(shufflenet, base_ch=200, groups=2)
shufflenet_g3 = wrapped_partial(shufflenet, base_ch=240, groups=3)
shufflenet_g4 = wrapped_partial(shufflenet, base_ch=272, groups=4)
shufflenet_g8 = wrapped_partial(shufflenet, base_ch=384, groups=8)


def shufflenet_arg_scope(is_training = True,
                    weight_decay = 0.0001,
                    batch_norm_decay = 0.997,
                    batch_norm_epsilon = 1e-5,
                    batch_norm_scale = True):

    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer = slim.l2_regularizer(weight_decay),
            weights_initializer = slim.variance_scaling_initializer(),
            activation_fn = tf.nn.relu,
            normalizer_fn = slim.batch_norm,
            normalizer_params = batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding = 'SAME') as arg_sc:
                return arg_sc



# if __name__ == "__main__":
#     inputs = tf.random_normal([1, 224, 224, 3])
#   
#     logits, end_points= shufflenet_g2(inputs,classes=1000)
#     print(end_points['Predictions'])   
#     writer = tf.summary.FileWriter("./logs_shufflenet", graph=tf.get_default_graph())
#     init = tf.global_variables_initializer()
#     with tf.Session() as sess:
#         sess.run(init)
#         pred = sess.run(end_points['Predictions'])
#         print(pred)
#         print(np.argmax(pred,1))
#         print(pred[:,np.argmax(pred,1)])
