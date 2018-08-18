# Tensorflow mandates these.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools

import tensorflow as tf

slim = tf.contrib.slim


def ShuffleNet(images, bottleneck_layer_size, is_training, reuse=False, shuffle=True, base_ch=144, groups=1):
    with tf.variable_scope('Stage1'):
        net = slim.conv2d(images, 24, [3, 3], 2)
        net = slim.max_pool2d(net, [3, 3], 2, padding='SAME')
    net = shuffle_stage(net, base_ch, 3, groups, is_training, 'Stage2')
    net = shuffle_stage(net, base_ch*2, 7, groups, is_training, 'Stage3')
    net = shuffle_stage(net, base_ch*4, 3, groups, is_training, 'Stage4')

    with tf.variable_scope('Stage5'):
        net = slim.dropout(net, 0.5, is_training=is_training)
        net = slim.conv2d(net, bottleneck_layer_size, [1, 1], stride=1,
                          padding="SAME",
                          scope="conv_stage5")
        net = slim.avg_pool2d(net, kernel_size=4, stride=1)
        net = tf.reduce_mean(net, [1, 2],  name="logits")

    return net, None


def shuffle_stage(net, output, repeat, group, is_training, scope="Stage"):
    with tf.variable_scope(scope):
        net = shuffle_bottleneck(net, output, 2, is_training, group, scope='Unit{}'.format(0))
        for i in range(repeat):
            net = shuffle_bottleneck(net, output, 1, is_training, group, scope='Unit{}'.format(i+1))
    return net


def shuffle_bottleneck(net, output, stride, is_training, group=1, scope="Unit"):
    if stride != 1:
        _b, _h, _w, _c = net.get_shape().as_list()
        output = output - _c

    with tf.variable_scope(scope):
        if stride != 1:
            net_skip = slim.avg_pool2d(net, [3, 3], stride, padding="SAME", scope="3x3AVGPool")
        else:
            net_skip = net

        net = group_conv(net, output, 1, group, is_training, relu=True, scope="1x1ConvIn")

        net = channel_shuffle(net, output, group, is_training, scope="ChannelShuffle")

        with tf.variable_scope("3x3DXConv"):
            depthwise_filter = tf.get_variable("depth_conv_w", [3, 3, output, 1],
                                               initializer=tf.truncated_normal_initializer(stddev=0.01))
            net = tf.nn.depthwise_conv2d(net, depthwise_filter, [1, stride, stride, 1], 'SAME', name="DWConv")

        net = group_conv(net, output, 1, group, is_training, relu=True, scope="1x1ConvOut")

        if stride != 1:
            net = tf.concat([net, net_skip], axis=3)
        else:
            net = net + net_skip

        net = tf.nn.relu(net)

    return net


def group_conv(net, output, stride, group, is_training, relu=True, scope="GConv"):
    num_channels_in_group = output//group
    with tf.variable_scope(scope):
        net = tf.split(net, group, axis=3, name="split")
        for i in range(group):
            net[i] = slim.conv2d(net[i], num_channels_in_group, [1, 1], stride=stride,
                                 activation_fn=tf.nn.relu if relu else None,
                                 normalizer_fn=slim.batch_norm,
                                 normalizer_params={'is_training':is_training})
        net = tf.concat(net, axis=3, name="concat")
    return net


def channel_shuffle(net, output, group, is_training, scope="ChannelShuffle"):
    num_channels_in_group = output//group
    with tf.variable_scope(scope):
        net = tf.split(net, output, axis=3, name="split")
        chs = []
        for i in range(group):
            for j in range(num_channels_in_group):
                chs.append(net[i + j * group])
        net = tf.concat(chs, axis=3, name="concat")
    return net


def shufflenet_arg_scope(is_training=True,
                           weight_decay=0.00005,
                           regularize_depthwise=False):
  """Defines the default MobilenetV2 arg scope.

  Args:
    is_training: Whether or not we're training the model.
    weight_decay: The weight decay to use for regularizing the model.
    regularize_depthwise: Whether or not apply regularization on depthwise.

  Returns:
    An `arg_scope` to use for the mobilenet v2 model.
  """
  batch_norm_params = {
      'is_training': is_training,
      'center': True,
      'scale': True,
      'fused': True,
      'decay': 0.995,
      'epsilon': 2e-5,
      # force in-place updates of mean and variance estimates
      'updates_collections': None,
      # Moving averages ends up in the trainable variables collection
      'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
  }

  # Set weight_decay for weights in Conv and InvResBlock layers.
  #weights_init = tf.truncated_normal_initializer(stddev=stddev)
  weights_init = tf.contrib.layers.xavier_initializer(uniform=False)
  regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  if regularize_depthwise:
    depthwise_regularizer = regularizer
  else:
    depthwise_regularizer = None
  with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                      weights_initializer=weights_init,
                      activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm): #tf.keras.layers.PReLU
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
        with slim.arg_scope([slim.separable_conv2d],
                            weights_regularizer=depthwise_regularizer) as sc:
          return sc


def inference(images, bottleneck_layer_size=128, phase_train=False,
              weight_decay=0.00005, reuse=False):
    '''build a mobilenet_v2 graph to training or inference.

    Args:
        images: a tensor of shape [batch_size, height, width, channels].
        bottleneck_layer_size: number of predicted classes. If 0 or None, the logits layer
          is omitted and the input features to the logits layer (before dropout)
          are returned instead.
        phase_train: Whether or not we're training the model.
        weight_decay: The weight decay to use for regularizing the model.
        reuse: whether or not the network and its variables should be reused. To be
          able to reuse 'scope' must be given.

    Returns:
        net: a 2D Tensor with the logits (pre-softmax activations) if bottleneck_layer_size
          is a non-zero integer, or the non-dropped-out input to the logits layer
          if bottleneck_layer_size is 0 or None.
        end_points: a dictionary from components of the network to the corresponding
          activation.

    Raises:
        ValueError: Input rank is invalid.
    '''
    # pdb.set_trace()
    arg_scope = shufflenet_arg_scope(is_training=phase_train, weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
        return ShuffleNet(images, bottleneck_layer_size=bottleneck_layer_size, is_training=phase_train, reuse=reuse)
    
    