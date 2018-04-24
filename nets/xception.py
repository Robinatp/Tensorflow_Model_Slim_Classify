
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
slim = tf.contrib.slim
import numpy as np

# =========================================================================== #
# Xception implementation (clean)
# =========================================================================== #
def xception(inputs,
             num_classes=1000,
             dropout_keep_prob=0.5,
             is_training=True,
             prediction_fn=slim.softmax,
             reuse=None,
             global_pool=True,
             scope='xception'):
    """Xception model from https://arxiv.org/pdf/1610.02357v2.pdf

    The default image size used to train_and_eval this network is 299x299.
    """

    # end_points collect relevant activations for external use, for example
    # summaries or losses.
    end_points = {}

    with tf.variable_scope(scope, 'xception', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points' 
        with slim.arg_scope([slim.conv2d, slim.separable_convolution2d, slim.max_pool2d],
                               outputs_collections = [end_points_collection]):
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                # Entry flow: blocks 1 to 4.
                with tf.variable_scope('entry_flow'):
                    # Block 1.
                    with tf.variable_scope('block1'):
                        net = slim.conv2d(inputs, 32, [3, 3], stride=2, padding='VALID', scope='conv1')
                        net = slim.conv2d(net, 64, [3, 3], padding='VALID', scope='conv2')
                    
            
                    # Residual block 2.
                    with tf.variable_scope('block2'):
                        res = slim.conv2d(net, 128, [1, 1], stride=2, activation_fn=None, scope='residual')
                        net = slim.separable_convolution2d(net, 128, [3, 3], 1, scope='sepconv1')
                        net = slim.separable_convolution2d(net, 128, [3, 3], 1, activation_fn=None, scope='sepconv2')
                        net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool')
                        net = res + net
                   
            
                    # Residual block 3.
                    with tf.variable_scope('block3'):
                        res = slim.conv2d(net, 256, [1, 1], stride=2, activation_fn=None, scope='residual')
                        net = tf.nn.relu(net)
                        net = slim.separable_convolution2d(net, 256, [3, 3], 1, scope='sepconv1')
                        net = slim.separable_convolution2d(net, 256, [3, 3], 1, activation_fn=None, scope='sepconv2')
                        net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool')
                        net = res + net
            
                    # Residual block 4.
                    with tf.variable_scope('block4'):
                        res = slim.conv2d(net, 728, [1, 1], stride=2, activation_fn=None, scope='residual')
                        net = tf.nn.relu(net)
                        net = slim.separable_convolution2d(net, 728, [3, 3], 1, scope='sepconv1')
                        net = slim.separable_convolution2d(net, 728, [3, 3], 1, activation_fn=None, scope='sepconv2')
                        net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool')
                        net = res + net
        
                # Middle flow blocks.
                with tf.variable_scope('middle_flow'):
                    for i in range(8):
                        end_point = 'block' + str(i + 5)
                        with tf.variable_scope(end_point):
                            res = net
                            net = tf.nn.relu(net)
                            net = slim.separable_convolution2d(net, 728, [3, 3], 1, activation_fn=None,
                                                               scope='sepconv1')
                            net = tf.nn.relu(net)
                            net = slim.separable_convolution2d(net, 728, [3, 3], 1, activation_fn=None,
                                                               scope='sepconv2')
                            net = tf.nn.relu(net)
                            net = slim.separable_convolution2d(net, 728, [3, 3], 1, activation_fn=None,
                                                               scope='sepconv3')
                            net = res + net
        
                # Exit flow: blocks 13 and 14.
                with tf.variable_scope('exit_flow'):
                    with tf.variable_scope('block13'):
                        res = slim.conv2d(net, 1024, [1, 1], stride=2, activation_fn=None, scope='residual')
                        net = tf.nn.relu(net)
                        net = slim.separable_convolution2d(net, 728, [3, 3], 1, activation_fn=None, scope='sepconv1')
                        net = tf.nn.relu(net)
                        net = slim.separable_convolution2d(net, 1024, [3, 3], 1, activation_fn=None, scope='sepconv2')
                        net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool')
                        net = res + net
            
                    with tf.variable_scope('block14'):
                        net = slim.separable_convolution2d(net, 1536, [3, 3], 1, scope='sepconv1')
                        net = slim.separable_convolution2d(net, 2048, [3, 3], 1, scope='sepconv2')
                    end_points[end_point] = net
                
                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                
                # Global averaging.
                with tf.variable_scope('Logits'):
                    if global_pool:
                        net = tf.reduce_mean(net, [1, 2], name='reduce_avg')

                    # dropout
                    if dropout_keep_prob:
                        net = slim.dropout(net,keep_prob=dropout_keep_prob, scope="dropout")
                        
                    if not num_classes:
                        return net, end_points
                    
                    logits = slim.fully_connected(net, num_classes, activation_fn=None)
        
                end_points['Logits'] = logits
                if prediction_fn:
                    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
        
            return logits, end_points

xception.default_image_size = 299


def xception_arg_scope(weight_decay=0.00001, stddev=0.1):
    """Defines the default Xception arg scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.
      stddev: The standard deviation of the trunctated normal weight initializer.

    Returns:
      An `arg_scope` to use for the xception model.
    """
    batch_norm_params = {
      # Decay for the moving averages.
      'decay': 0.9997,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.separable_convolution2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
                [slim.conv2d, slim.separable_convolution2d],
                padding='SAME',
                weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as sc:
                return sc
            
            
if __name__ == "__main__":
    inputs = tf.random_normal([1, 299, 299, 3])
    with slim.arg_scope(xception_arg_scope()):
         logits, end_points= xception(inputs,1000)
      
    writer = tf.summary.FileWriter("./logs_xception", graph=tf.get_default_graph())
    print("Layers")
    for k, v in end_points.items():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))
      
    print("Parameters")
    for v in slim.get_model_variables():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))
          
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        pred = sess.run(end_points['Predictions'])
#         print(pred)
        print(np.argmax(pred,1))
#         print(pred[:,np.argmax(pred,1)])



