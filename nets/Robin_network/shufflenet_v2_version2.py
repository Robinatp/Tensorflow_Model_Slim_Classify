from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

slim = tf.contrib.slim


def _channel_shuffle(inputs, num_groups, scope=None):
    if num_groups == 1:
        return inputs
    with tf.variable_scope(scope, 'channel_shuffle', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        assert depth_in % num_groups == 0, (
            "depth_in=%d is not divisible by num_groups=%d" %
            (depth_in, num_groups))
        # group size, depth = g * n
        group_size = depth_in // num_groups
        net = inputs
        net_shape = net.shape.as_list()
        # reshape to (b, h, w, g, n)
        net = tf.reshape(net, net_shape[:3] + [num_groups, group_size])
        # transpose to (b, h, w, n, g)
        net = tf.transpose(net, [0, 1, 2, 4, 3])
        # reshape back to (b, h, w, depth)
        net = tf.reshape(net, net_shape)
        return net


@slim.add_arg_scope
def stack_blocks_dense(net, blocks, output_stride=None,
                       outputs_collections=None):
    # The current_stride variable keeps track of the effective stride of the
    # activations. This allows us to invoke atrous convolution whenever applying
    # the next residual unit would result in the activations having stride larger
    # than the target output_stride.
    current_stride = 1
    
    # The atrous convolution rate parameter.
    rate = 1
    
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError('The target output_stride cannot be reached.')
                
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    # If we have reached the target output_stride, then we need to employ
                    # atrous convolution with stride=1 and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
                        rate *= unit.get('stride', 1)
                    
                    else:
                        net = block.unit_fn(net, rate=1, **unit)
                        current_stride *= unit.get('stride', 1)
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
    
    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')
    
    return net

@slim.add_arg_scope
def shufflenet_v2_unit(inputs,
                       depth,
                       stride,
                       groups,
                       left_ratio,
                       spatial_down,
                       rate=1,
                       outputs_collections=None,
                       scope=None,
                       use_bounded_activations=False):
    with tf.variable_scope(scope, 'bottleneck', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)

        if spatial_down:
            
            depth_left = int(depth * left_ratio)    # left branch depth
            depth = depth - depth_left    # right branch depth
            
            # construct left branch
            # 3x3 depthwise conv. By passing filters=None
            # separable_conv2d produces only a depthwise convolution layer
            shortcut = slim.separable_conv2d(inputs, None, [3, 3],
                                             depth_multiplier=1,
                                             stride=stride,
                                             rate=rate,
                                             activation_fn=None,
                                             scope='shortcut_3X3_DWConv')
            # 1x1 conv
            shortcut = slim.conv2d(shortcut, depth, 1, stride=1, padding='SAME', scope='shortcut_1X1_Conv')
            residual = inputs
        else:
            depth_left = int(depth * left_ratio)    # left branch depth
            depth = depth - depth_left    # right branch depth
            shortcut, residual = tf.split(inputs, [depth_left, depth], axis=-1,name="channel_split")

        # construct right branch
        # 1x1 conv
        residual = slim.conv2d(residual, depth, 1, stride=1, padding='SAME', scope='1X1_ConvIn')
        # 3x3 depthwise conv. By passing filters=None
        # separable_conv2d produces only a depthwise convolution layer
        residual = slim.separable_conv2d(residual, None, [3, 3],
                                         depth_multiplier=1,
                                         stride=stride,
                                         rate=rate,
                                         activation_fn=None,
                                         scope='3X3_DWConv')
        # 1x1 conv
        residual = slim.conv2d(residual, depth, 1, stride=1, padding='SAME', scope='1X1_ConvOut')
        # concat two branches
        output = tf.concat([shortcut, residual], axis=-1)
        # channel shuflle
        output = _channel_shuffle(output, num_groups=groups, scope='channel_shuffle')
        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                output)


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.
   
    Its parts are:
      scope: The scope of the `Block`.
      unit_fn: The ShuffleNet V2 unit function which takes as input a `Tensor` and
        returns another `Tensor` with the output of the ShuffleNet V2 unit.
      args: A list of length equal to the number of units in the `Block`. The list
        contains one (depth, stride, groups, left_ratio) tuple for each unit in the
        block to serve as argument to unit_fn.
    """


def shufflenet_v2_block(scope, depth, num_units, stride, groups=2, left_ratio=0.5):
    """Helper function for creating a shufflenet v2 bottleneck block.
   
    Args:
      scope: The scope of the block.
      depth: The depth of the bottleneck layer for each unit.
      num_units: The number of units in the block.
      stride: The stride of the block, implemented as a stride in the last unit.
        All other units have stride=1.
      groups: number of groups for each unit except the first unit.
      left_ratio: split channel ratio for left branch.
   
    Returns:
      A shufflenet v2 bottleneck block.
    """
    return Block(scope, shufflenet_v2_unit, [{
        'depth': depth,
        'stride': stride,
        'groups': groups,
        'spatial_down': True,
        'left_ratio': left_ratio
    }] + [{
        'depth': depth,
        'stride': 1,
        'groups': groups,
        'spatial_down': False,
        'left_ratio': left_ratio
    }]  * (num_units - 1))


def shufflenet_v2_base(inputs,
                       blocks,
                       output_channels=1024,
                       num_classes=None,
                       is_training=True,
                       global_pool=True,
                       output_stride=None,
                       include_root_block=True,
                       spatial_squeeze=True,
                       dropout_keep_prob=None,
                       reuse=None,
                       scope=None):
    with tf.variable_scope(scope, 'Shufflenet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, shufflenet_v2_unit, stack_blocks_dense,
                             slim.separable_conv2d],
                            outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                end_points = {}
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    net = slim.conv2d(inputs, 24, 3, stride=2, padding='SAME', scope='conv1')
                    net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='pool1')
                else:
                    height = inputs.get_shape()[1]
                    stride = 2 if height > 32 else 1
                    net = slim.conv2d(net, 24, 3, stride=stride, scope='conv1')
            
                net = stack_blocks_dense(net, blocks, output_stride)
                # final 1x1 conv
                net = slim.conv2d(net, output_channels, 1, stride=1, scope='conv5')
                if global_pool:
                    # Global average pooling.
                    net = tf.reduce_mean(net, [1, 2], name='pool5', keepdims=True)
                    
                if num_classes is not None:
                    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                      normalizer_fn=None, scope='logits')
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
            
                # Convert end_points_collection into a dictionary of end_points.
                end_points.update(slim.utils.convert_collection_to_dict(
                    end_points_collection))
                if num_classes is not None:
                    end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points
shufflenet_v2_base.default_image_size = 224


# shufflenet v2 depths dictionary
depths_dict = {0.5: (48, 96, 192, 1024),
               1.0: (116, 232, 464, 1024),
               1.5: (176, 352, 704, 1024),
               2.0: (244, 488, 976, 2048)}


def shufflenet_v2(inputs,
                  num_classes=None,
                  dropout_keep_prob=None,
                  is_training=True,
                  depth_multiplier=1.0,
                  min_depth=8,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='Shufflenet_v2'):
    """Shufflenet v2."""
    depths = depths_dict[depth_multiplier]
    blocks =  [
        shufflenet_v2_block('Stage2', depth=depths[0], num_units=4,
                            stride=2, groups=2, left_ratio=0.5),
        shufflenet_v2_block('Stage3', depth=depths[1], num_units=8,
                            stride=2, groups=2, left_ratio=0.5),
        shufflenet_v2_block('Stage4', depth=depths[2], num_units=4,
                            stride=2, groups=2, left_ratio=0.5),
    ]
    return shufflenet_v2_base(inputs, blocks,
                              output_channels=depths[3],
                              num_classes=num_classes,
                              is_training=is_training,
                              global_pool=global_pool,
                              output_stride=output_stride,
                              include_root_block=True,
                              spatial_squeeze=spatial_squeeze,
                              dropout_keep_prob=dropout_keep_prob,
                              reuse=reuse,
                              scope=scope)
shufflenet_v2.default_image_size = shufflenet_v2_base.default_image_size


def shufflenet_v2_arg_scope(is_training=True,
                            weight_decay=0.00004,
                            stddev=0.09,
                            regularize_depthwise=False):
    """Defines the default ShufflenetV2 arg scope.
    
    Args:
      is_training: Whether or not we're training the model.
      weight_decay: The weight decay to use for regularizing the model.
      stddev: The standard deviation of the trunctated normal weight initializer.
      regularize_depthwise: Whether or not apply regularization on depthwise.
    
    Returns:
      An `arg_scope` to use for the mobilenet v1 model.
    """
    batch_norm_params = {
        'is_training': is_training,
        'center': True,
        'scale': True,
        'decay': 0.9997,
        'epsilon': 0.001,
    }
    
    # Set weight_decay for weights in Conv and DepthSepConv layers.
    weights_init = slim.variance_scaling_initializer()
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_initializer=weights_init,
                        activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
                with slim.arg_scope([slim.separable_conv2d],
                                    weights_regularizer=depthwise_regularizer) as sc:
                    return sc
                


if __name__ == "__main__":
    inputs = tf.random_normal([1, 224, 224, 3])
    
    with slim.arg_scope(shufflenet_v2_arg_scope(is_training = False)):
        logits ,end_points= shufflenet_v2(inputs, num_classes=1000,is_training=False);
   
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
#         pred = sess.run(end_points['predictions'])
#         print(pred)
#         print(np.argmax(pred,1))
#         print(pred[:,np.argmax(pred,1)])
