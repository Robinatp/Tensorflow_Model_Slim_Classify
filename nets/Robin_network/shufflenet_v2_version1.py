import tensorflow as tf
import tensorflow.contrib.slim as slim


BATCH_NORM_MOMENTUM = 0.997
BATCH_NORM_EPSILON = 1e-3


def shufflenet(images, is_training, num_classes=1000, depth_multiplier='1.0'):
    """
    This is an implementation of ShuffleNet v2:
    https://arxiv.org/abs/1807.11164

    Arguments:
        images: a float tensor with shape [batch_size, image_height, image_width, 3],
            a batch of RGB images with pixel values in the range [0, 1].
        is_training: a boolean.
        num_classes: an integer.
        depth_multiplier: a string, possible values are '0.5', '1.0', '1.5', and '2.0'.
    Returns:
        a float tensor with shape [batch_size, num_classes].
    """
    possibilities = {'0.5': 48, '1.0': 116, '1.5': 176, '2.0': 224}
    initial_depth = possibilities[depth_multiplier]

    def batch_norm(x):
        x = tf.layers.batch_normalization(
            x, axis=3, center=True, scale=True,
            training=is_training,
            momentum=BATCH_NORM_MOMENTUM,
            epsilon=BATCH_NORM_EPSILON,
            fused=True, name='batch_norm'
        )
        return x

    with tf.name_scope('standardize_input'):
        net = (2.0 * images) - 1.0

    with tf.variable_scope('ShuffleNetV2'):
        params = {
            'padding': 'SAME', 'activation_fn': tf.nn.relu,
            'normalizer_fn': batch_norm, 'data_format': 'NHWC',
            'weights_initializer': tf.contrib.layers.xavier_initializer()
        }
        with slim.arg_scope([slim.conv2d,slim.separable_conv2d], **params):

            net = slim.conv2d(net, 24, (3, 3), stride=2, scope='Conv1')
            net = slim.max_pool2d(net, (3, 3), stride=2, padding='SAME', scope='MaxPool')
            
            net = shuffle_stage(net,  3, out_channels=initial_depth, scope="Stage2")
            net = shuffle_stage(net,  7, out_channels=initial_depth, scope="Stage3")
            net = shuffle_stage(net,  3, out_channels=initial_depth, scope="Stage4")

#             with tf.variable_scope('Stage2'):
#                 x, y = shuffle_bottleneck_unit_with_downsampling(x, out_channels=initial_depth)
#                 for j in range(3):
#                     with tf.variable_scope('unit_%d' % (j + 2)):
#                         x, y = concat_shuffle_split(x, y)
#                         x = shuffle_bottleneck_unit(x)
#                 x = tf.concat([x, y], axis=3)
# 
#             with tf.variable_scope('Stage3'):
#                 x, y = shuffle_bottleneck_unit_with_downsampling(x)
#                 for j in range(7):
#                     with tf.variable_scope('unit_%d' % (j + 2)):
#                         x, y = concat_shuffle_split(x, y)
#                         x = shuffle_bottleneck_unit(x)
#                 x = tf.concat([x, y], axis=3)
# 
#             with tf.variable_scope('Stage4'):
#                 x, y = shuffle_bottleneck_unit_with_downsampling(x)
#                 for j in range(3):
#                     with tf.variable_scope('unit_%d' % (j + 2)):
#                         x, y = concat_shuffle_split(x, y)
#                         x = shuffle_bottleneck_unit(x)
#                 x = tf.concat([x, y], axis=3)

            final_channels = 1024 if depth_multiplier != '2.0' else 2048
            net = slim.conv2d(net, final_channels, (1, 1), stride=1, scope='Conv5')

    # global average pooling
    net = tf.reduce_mean(net, axis=[1, 2])

    logits = slim.fully_connected(
        net, num_classes, activation_fn=None, scope='classifier',
        weights_initializer=tf.contrib.layers.xavier_initializer()
    )
    return logits


def concat_shuffle_split(x, y):
    with tf.name_scope('concat_shuffle_split'):
        shape = tf.shape(x)
        batch_size = shape[0]
        height, width = shape[1], shape[2]
        depth = x.shape[3].value

        z = tf.stack([x, y], axis=3)  # shape [batch_size, height, width, 2, depth]
        z = tf.transpose(z, [0, 1, 2, 4, 3])
        z = tf.reshape(z, [batch_size, height, width, 2*depth])
        x, y = tf.split(z, num_or_size_splits=2, axis=3)
        return x, y


def shuffle_bottleneck_unit(x):
    in_channels = x.shape[3].value
    residual = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='1x1ConvIn')

    # separable_conv2d produces only a depthwise convolution layer
    residual = slim.separable_conv2d(residual, None, [3, 3],
                                         depth_multiplier=1,
                                         stride=1,
                                         rate=1,
                                         activation_fn=None,
                                         scope='DWConv')
    
    residual = slim.conv2d(residual, in_channels, (1, 1), stride=1, scope='1x1ConvOut')
    return residual


def shuffle_bottleneck_unit_with_downsampling(inputs, out_channels=None):
    with tf.variable_scope('unit_1'):
        in_channels = inputs.shape[3].value
        out_channels = 2 * in_channels if out_channels is None else out_channels

        residual = slim.conv2d(inputs, in_channels, (1, 1), stride=1, scope='1x1ConvIn')
        
        # separable_conv2d produces only a depthwise convolution layer
        residual = slim.separable_conv2d(residual, None, [3, 3],
                                         depth_multiplier=1,
                                         stride=2,
                                         rate=1,
                                         activation_fn=None,
                                         scope='DWConv')
        
        
        residual = slim.conv2d(residual, out_channels // 2, (1, 1), stride=1, scope='1x1ConvOut')

        with tf.variable_scope('shortcut'):
            shortcut = slim.separable_conv2d(inputs, None, [3, 3],
                                         depth_multiplier=1,
                                         stride=2,
                                         rate=1,
                                         activation_fn=None,
                                         scope='DWConv')
            shortcut = slim.conv2d(shortcut, out_channels // 2, (1, 1), stride=1, scope='1x1ConvOut')
            return shortcut, residual


def shuffle_stage(inputs,  repeat, out_channels=None, scope=None): 
    with tf.variable_scope(scope,"Stage",[inputs]) as sc:
        x, y = shuffle_bottleneck_unit_with_downsampling(inputs, out_channels)
        for j in range(repeat):
             with tf.variable_scope('unit_%d' % (j + 2)):
                 x, y = concat_shuffle_split(x, y)
                 x = shuffle_bottleneck_unit(x)
        output = tf.concat([x, y], axis=3)
        return output
                
    


if __name__ == "__main__":
    inputs = tf.random_normal([1, 224, 224, 3])
    
    logits = shufflenet(inputs, is_training = False, num_classes=1000, depth_multiplier='1.0')
   
    writer = tf.summary.FileWriter("./logs", graph=tf.get_default_graph())
    
    
    print("Parameters")
    for v in slim.get_model_variables():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))  
    
    
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
#         pred = sess.run(end_points['predictions'])
#         print(pred)
#         print(np.argmax(pred,1))
#         print(pred[:,np.argmax(pred,1)])
