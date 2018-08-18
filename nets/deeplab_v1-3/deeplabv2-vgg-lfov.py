from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
slim = tf.contrib.slim


def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.
  Args:
    weight_decay: The l2 regularization coefficient.
  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d], padding='SAME') as arg_sc:
      return arg_sc
  
def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=True):
  """Oxford Net VGG 16-Layers version D Example.
  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)
  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d, slim.avg_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],rate=2, scope='conv5')#atrous convolution X3
      net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')
      net = slim.avg_pool2d(net, [3, 3], stride=1, scope='avg_pool5')

      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 1024, [3,3], rate=12, scope='fc6')#atrous convolution
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
      net = slim.conv2d(net, 1024, [1, 1], scope='fc7')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
      
      
      net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8_voc_21')#softmax
      
      with tf.variable_scope("upsample"):   
        raw_output = tf.image.resize_bilinear(net, tf.shape(inputs)[1:3,])
        raw_output = tf.argmax(raw_output, axis=3)
        raw_output = tf.expand_dims(raw_output, axis=3) # Create 4D-tensor.
        logit = tf.cast(raw_output, tf.uint8)
      
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keepdims=True, name='global_pool')
        end_points['global_pool'] = net
          
        
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='squeezed')
        end_points["squeezed"] = net
        
      return logit, end_points
  

vgg_16.default_image_size = 224

if __name__ == "__main__":
    inputs = tf.random_normal([1, 321, 321, 3])
    with slim.arg_scope(vgg_arg_scope()):
        outputs, end_points = vgg_16(inputs,21,is_training=False)
      
    writer = tf.summary.FileWriter("./log", graph=tf.get_default_graph())
    print("Layers")
    for k, v in end_points.items():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))
      
    print("Parameters")
    for v in slim.get_model_variables():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))
          
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        pred = sess.run(end_points["squeezed"])
#         print(pred)
        print(np.argmax(pred,1))
#         print(pred[:,np.argmax(pred,1)])

