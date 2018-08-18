"""SqueezeNet.

here is SqueezeNet structure.

"""

# Tensorflow mandates these.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools

import tensorflow as tf

slim = tf.contrib.slim


def SqueezeNet(images, bottleneck_layer_size, is_training):
    # conv1
    net = slim.conv2d(images, 96, [7, 7], stride=2,
                      padding="SAME", scope="conv1")
    # maxpool1
    net = slim.max_pool2d(net, [3, 3], stride=2, scope="maxpool1")

    # fire2
    net = fire(net, 16, 64, "fire2")
    # fire3
    net = fire(net, 16, 64, "fire3")
    # fire4
    net = fire(net, 32, 128, "fire4")

    # maxpool4
    net = slim.max_pool2d(net, [3, 3], stride=2, scope="maxpool4")

    # fire5
    net = fire(net, 32, 128, "fire5")
    # fire6
    net = fire(net, 48, 192, "fire6")
    # fire7
    net = fire(net, 48, 192, "fire7")
    # fire8
    net = fire(net, 64, 256, "fire8")

    # maxpool8
    net = slim.max_pool2d(net, [3, 3], stride=2, scope="maxpool8")

    # fire9
    net = fire(net, 64, 256, "fire9")

    # droupout
    net = slim.dropout(net, 0.5, is_training=is_training)
    # conv10
    net = slim.conv2d(net, bottleneck_layer_size, [1, 1], stride=1,
                      padding="SAME",
                      scope="conv10")
    # avgpool10
    net = slim.avg_pool2d(net, kernel_size=6, stride=1)
    # squeeze the axis
    net = tf.squeeze(net, axis=[1, 2], name="logits")

    return net, None


def fire(input, squeeze_depth, expand_depth, scope):
    with tf.variable_scope(scope):
        squeeze = slim.conv2d(input, squeeze_depth, [1, 1],
                              stride=1, padding="SAME", scope="squeeze")
        # squeeze
        expand_1x1 = slim.conv2d(squeeze, expand_depth, [1, 1],
                                 stride=1, padding="SAME", scope="expand_1x1")
        expand_3x3 = slim.conv2d(squeeze, expand_depth, [3, 3],
                                 padding="SAME", scope="expand_3x3")
        return tf.concat([expand_1x1, expand_3x3], axis=3)


def squeezenet_arg_scope(is_training=True,
                           weight_decay=0.00005):
  """Defines the default squeezenet arg scope.

  Args:
    is_training: Whether or not we're training the model.
    weight_decay: The weight decay to use for regularizing the model.

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
      'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES ],
  }

  weights_init = tf.contrib.layers.xavier_initializer(uniform=False)
  regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

  with slim.arg_scope([slim.conv2d],
                      weights_initializer=weights_init,
                      activation_fn=tf.nn.elu, normalizer_fn=slim.batch_norm):  # tf.keras.layers.PReLU
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer) as sc:
          return sc


def inference(images, bottleneck_layer_size=128, phase_train=False,
              weight_decay=0.00005):
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
    arg_scope = squeezenet_arg_scope(is_training=phase_train, weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
        return SqueezeNet(images, bottleneck_layer_size=bottleneck_layer_size, is_training=phase_train)