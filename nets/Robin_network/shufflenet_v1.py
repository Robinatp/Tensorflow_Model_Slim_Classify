# Tensorflow mandates these.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
import time
from datetime import datetime


slim = tf.contrib.slim


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.
   
    Its parts are:
      scope: The scope of the `Block`.
      unit_fn: The ShuffleNet unit function which takes as input a `Tensor` and
        returns another `Tensor` with the output of the ShuffleNet unit.
      args: A list of length equal to the number of units in the `Block`. The list
        contains one (depth, depth_bottleneck, stride, groups) tuple for each unit in the
        block to serve as argument to unit_fn.
    """


def channel_shuffle(inputs, num_groups, scope=None):
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
        n, h, w, c = net.shape.as_list()
        # reshape to (b, h, w, g, n)
        net = tf.reshape(net, [-1,h,w,num_groups, group_size])
        # transpose to (b, h, w, n, g)
        net = tf.transpose(net, [0, 1, 2, 4, 3])
        # reshape back to (b, h, w, depth)
        net = tf.reshape(net, [-1,h,w,c])
        return net
 
 
    

@slim.add_arg_scope
def group_conv2d(inputs, num_outputs, kernel_size, num_groups=1,
                 stride=1, rate=1, padding='SAME',
                 activation_fn=tf.nn.relu,
                 normalizer_fn=None,
                 normalizer_params=None,
                 biases_initializer=tf.zeros_initializer(),
                 scope=None):
    with tf.variable_scope(scope, 'group_conv2d', [inputs]) as sc:
        biases_initializer = biases_initializer if normalizer_fn is None else None
        if num_groups == 1:
            return slim.conv2d(inputs, num_outputs, kernel_size,
                               stride=stride, rate=rate,
                               padding=padding,
                               activation_fn=activation_fn,
                               normalizer_fn=normalizer_fn,
                               normalizer_params=normalizer_params,
                               biases_initializer=biases_initializer,
                               scope=scope)
        else:
            depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)

            assert num_outputs % num_groups == 0, (
                "num_outputs=%d is not divisible by num_groups=%d" %
                (num_outputs, num_groups))
            assert depth_in % num_groups == 0, (
                "depth_in=%d is not divisible by num_groups=%d" %
                (depth_in, num_groups))

            group_size_out = num_outputs // num_groups
            input_slices = tf.split(inputs, num_groups, axis=-1)
            output_slices = [slim.conv2d(inputs=input_slice,
                                         num_outputs=group_size_out,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         rate=rate,
                                         padding=padding,
                                         activation_fn=None,
                                         normalizer_fn=None,
                                         biases_initializer=biases_initializer,
                                         scope=scope + '/group%d' % idx)
                             for idx, input_slice in enumerate(input_slices)]
            net = tf.concat(output_slices, axis=-1)

            if normalizer_fn is not None:
                normalizer_params = normalizer_params or {}
                net = normalizer_fn(net, **normalizer_params)
            if activation_fn is not None:
                net = activation_fn(net)
            return net


@slim.add_arg_scope
def shufflenet_unit(inputs,
                    depth,
                    depth_bottleneck,
                    stride,
                    groups,
                    rate=1,
                    outputs_collections=None,
                    scope=None,
                    use_bounded_activations=False):
    with tf.variable_scope(scope, 'bottleneck', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        if stride == 2:
            ratio = depth // depth_bottleneck
            depth -= depth_in
            depth_bottleneck = depth // ratio
            depth = depth_bottleneck * ratio
            
        # 1x1 group conv
        residual = group_conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                                num_groups=groups, scope='1x1GConvIn')
        # channel shuffle
        residual = channel_shuffle(residual, groups, 'channel_shuffle')
        # 3x3 depthwise conv. By passing filters=None
        # separable_conv2d produces only a depthwise convolution layer
        residual = slim.separable_conv2d(residual, None, [3, 3],
                                         depth_multiplier=1,
                                         stride=stride,
                                         rate=rate,
                                         activation_fn=None,
                                         scope='DWConv')
        residual = group_conv2d(residual, depth, [1, 1], stride=1,
                                num_groups=groups, activation_fn=None,
                                scope='1x1GConvOut')
        if stride == 1:
            shortcut = inputs
            output = tf.nn.relu(shortcut + residual)
        else:
            shortcut = slim.avg_pool2d(inputs, [3, 3], stride=2, scope='pool1',
                                       padding='SAME')
            output = tf.nn.relu(tf.concat([shortcut, residual], axis=3))
        
        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                output)

def shufflenet_block(scope, base_depth, num_units, stride, groups, groups_in=None):
    """Helper function for creating a shufflenet bottleneck block.
   
    Args:
      scope: The scope of the block.
      base_depth: The depth of the bottleneck layer for each unit.
      num_units: The number of units in the block.
      stride: The stride of the block, implemented as a stride in the last unit.
        All other units have stride=1.
      groups_in: number of groups for first unit.
      groups: number of groups for each unit except the first unit.
   
    Returns:
      A shufflenet bottleneck block.
    """
    return Block(scope, shufflenet_unit, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth * 3,#control the output or input of the depth_bottleneck and DWConv
        'stride': stride,
        'groups': groups if groups_in is None else groups_in#groups_in=1
    }] + [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth * 3,
        'stride': 1,
        'groups': groups
    }]  * (num_units - 1))


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


def shufflenet_base(inputs,
                    blocks,
                    num_classes=None,
                    is_training=True,
                    global_pool=True,
                    output_stride=None,
                    include_root_block=True,
                    spatial_squeeze=True,
                    dropout_keep_prob=None,
                    reuse=None,
                    scope=None):
#     with tf.variable_scope(scope, 'Shufflenet_v1', [inputs], reuse=reuse) as sc:
        end_points_collection = '_end_points'
        with slim.arg_scope([slim.conv2d, shufflenet_unit, stack_blocks_dense,
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
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                else:
                    height = inputs.get_shape()[1]
                    stride = 2 if height > 32 else 1
                    net = slim.conv2d(net, 24, 3, stride=stride, scope='conv1')
            
                net = stack_blocks_dense(net, blocks, output_stride)
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
shufflenet_base.default_image_size = 224


def shufflenet_v1(inputs,
                  num_classes=1000,
                  dropout_keep_prob=None,
                  is_training=True,
                  depth_multiplier=1.0,
                  min_depth=8,
                  groups=3,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='Shufflenet_v1'):
    """Shufflenet."""
    Depth_Channels = {'1': [144, 288, 576], '2': [200, 400, 800], '3': [240, 480, 960],
                      '4': [272, 544, 1088], '8': [384, 768, 1536],}
    groups_str = str(groups)
    assert groups_str in ['1', '2', '3', '4', '8'], (
        'groups must be one of [1, 2, 3, 4, 8], your groups=%d' % groups)
    depth_multi = lambda d: max(int(d * depth_multiplier), min_depth)
    depths = [depth_multi(depth) for depth in Depth_Channels[groups_str]]
    base_depths = [depth // 4 for depth in depths]# use for decrease depth_bottleneck 
    blocks =  [
        # we do not apply group convolution on the first pointwise layer beause the number
        # of input channels is relatively small
        shufflenet_block('Stage2', base_depth=base_depths[0], num_units=4, stride=2, groups=groups, groups_in=1),#groups_in=1
        shufflenet_block('Stage3', base_depth=base_depths[1], num_units=8, stride=2, groups=groups),
        shufflenet_block('Stage4', base_depth=base_depths[2], num_units=4, stride=2, groups=groups),
    ]

    return shufflenet_base(inputs, blocks,
                           num_classes, is_training,
                           global_pool=global_pool,
                           output_stride=output_stride,
                           include_root_block=True,
                           spatial_squeeze=spatial_squeeze,
                           dropout_keep_prob=dropout_keep_prob,
                           reuse=reuse,
                           scope=scope)
    
shufflenet_v1.default_image_size = shufflenet_base.default_image_size


def shufflenet_v1_arg_scope(is_training=True,
                            weight_decay=0.00004,
                            stddev=0.09,
                            regularize_depthwise=False):
    """Defines the default ShufflenetV1 arg scope.
    
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
        with slim.arg_scope([group_conv2d], activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
                    with slim.arg_scope([slim.separable_conv2d],
                                        weights_regularizer=depthwise_regularizer) as sc:
                        return sc
                    
                    
                    



def shufflenet_v1_base(inputs,
                  num_classes=1000,
                  dropout_keep_prob=None,
                  is_training=True,
                  depth_multiplier=1.0,
                  min_depth=8,
                  groups=3,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='Shufflenet_v1'):
    """Shufflenet."""
    Depth_Channels = {'1': [144, 288, 576], '2': [200, 400, 800], '3': [240, 480, 960],
                      '4': [272, 544, 1088], '8': [384, 768, 1536],}
    groups_str = str(groups)
    assert groups_str in ['1', '2', '3', '4', '8'], (
        'groups must be one of [1, 2, 3, 4, 8], your groups=%d' % groups)
    depth_multi = lambda d: max(int(d * depth_multiplier), min_depth)
    depths = [depth_multi(depth) for depth in Depth_Channels[groups_str]]
    base_depths = [depth // 4 for depth in depths]# use for decrease depth_bottleneck 
    blocks =  [
        # we do not apply group convolution on the first pointwise layer beause the number
        # of input channels is relatively small
        shufflenet_block('Stage2', base_depth=base_depths[0], num_units=4, stride=2, groups=groups, groups_in=1),#groups_in=1
        shufflenet_block('Stage3', base_depth=base_depths[1], num_units=8, stride=2, groups=groups),
        shufflenet_block('Stage4', base_depth=base_depths[2], num_units=4, stride=2, groups=groups),
    ]
    
    include_root_block=True
    
    end_points_collection = '_end_points'
    with slim.arg_scope([slim.conv2d, shufflenet_unit, stack_blocks_dense,
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
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            else:
                height = inputs.get_shape()[1]
                stride = 2 if height > 32 else 1
                net = slim.conv2d(net, 24, 3, stride=stride, scope='conv1')
        
            net = stack_blocks_dense(net, blocks, output_stride)
            
            # Convert end_points_collection into a dictionary of end_points.
            end_points.update(slim.utils.convert_collection_to_dict(
                end_points_collection))
          
            return end_points


class NoOpScope(object):
  """No-op context manager."""

  def __enter__(self):
    return None

  def __exit__(self, exc_type, exc_value, traceback):
    return False


def safe_arg_scope(funcs, **kwargs):
  """Returns `slim.arg_scope` with all None arguments removed.

  Arguments:
    funcs: Functions to pass to `arg_scope`.
    **kwargs: Arguments to pass to `arg_scope`.

  Returns:
    arg_scope or No-op context manager.

  Note: can be useful if None value should be interpreted as "do not overwrite
    this parameter value".
  """
  filtered_args = {name: value for name, value in kwargs.items()
                   if value is not None}
  if filtered_args:
    return slim.arg_scope(funcs, **filtered_args)
  else:
    return NoOpScope()

                    
                    
def training_scope(is_training=True,
                   weight_decay=0.00004,
                   stddev=0.09,
                   dropout_keep_prob=0.8,
                   bn_decay=0.997):
  """Defines Mobilenet training scope.

  Usage:
     with tf.contrib.slim.arg_scope(mobilenet.training_scope()):
       logits, endpoints = mobilenet_v2.mobilenet(input_tensor)

     # the network created will be trainble with dropout/batch norm
     # initialized appropriately.
  Args:
    is_training: if set to False this will ensure that all customizations are
      set to non-training mode. This might be helpful for code that is reused
      across both training/evaluation, but most of the time training_scope with
      value False is not needed. If this is set to None, the parameters is not
      added to the batch_norm arg_scope.

    weight_decay: The weight decay to use for regularizing the model.
    stddev: Standard deviation for initialization, if negative uses xavier.
    dropout_keep_prob: dropout keep probability (not set if equals to None).
    bn_decay: decay for the batch norm moving averages (not set if equals to
      None).

  Returns:
    An argument scope to use via arg_scope.
  """
  # Note: do not introduce parameters that would change the inference
  # model here (for example whether to use bias), modify conv_def instead.
  batch_norm_params = {
      'decay': bn_decay,
      'is_training': is_training
  }
  if stddev < 0:
    weight_intitializer = slim.initializers.xavier_initializer()
  else:
    weight_intitializer = tf.truncated_normal_initializer(stddev=stddev)

  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected, slim.separable_conv2d],
      weights_initializer=weight_intitializer,
      normalizer_fn=slim.batch_norm), \
      safe_arg_scope([slim.batch_norm], **batch_norm_params), \
      safe_arg_scope([slim.dropout], is_training=is_training,
                     keep_prob=dropout_keep_prob), \
      slim.arg_scope([slim.conv2d], \
                     weights_regularizer=slim.l2_regularizer(weight_decay)), \
      slim.arg_scope([slim.separable_conv2d], weights_regularizer=None) as s:
    return s

if __name__ == "__main__":
    inputs = tf.random_normal([1, 300, 300, 3])
    with slim.arg_scope(shufflenet_v1_arg_scope(is_training = False)):
        logits ,end_points= shufflenet_v1(inputs, num_classes=1000,is_training=False);
   
    writer = tf.summary.FileWriter("./logs", graph=tf.get_default_graph())
    
    
    print("Layers")
    for k, v in end_points.items():
        print('k={}, name = {}, shape = {}'.format(k, v.name, v.get_shape()))
    
    print("Parameters")
    for v in slim.get_model_variables():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))  
    
    
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        
        for i in range(10):
            start_time = time.time()
            pred = sess.run(logits)
            duration = time.time() - start_time
            print ('%s: step %d, duration = %.3f' %(datetime.now(), i, duration))