import collections
import tensorflow as tf
import time
from datetime import datetime

slim = tf.contrib.slim


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """A named tuple describing an Xception block.

  Its parts are:
    scope: The scope of the block.
    unit_fn: The Xception unit function which takes as input a tensor and
      returns another tensor with the output of the Xception unit.
    args: A list of length equal to the number of units in the block. The list
      contains one dictionary for each unit in the block to serve as argument to
      unit_fn.
  """

@slim.add_arg_scope
def res_block(inputs,
                 bottleneck_width=256,
                 outputs_collections=None,
                 scope=None):
        with tf.variable_scope(scope, 'dense_block', [inputs]) as sc:
            inter_channel = int(bottleneck_width/2)
            
            # left channel
            conv_right_0 = slim.conv2d(inputs, inter_channel, [1,1], stride=1, scope='conv_right_0')
            conv_right_1 = slim.conv2d(conv_right_0, inter_channel, [3,3], stride=1, scope='conv_right_1')
            conv_right_2 = slim.conv2d(conv_right_1, bottleneck_width, [3,3], stride=1, scope='conv_right_2')
            
            # right channel
            conv_right_0 = slim.conv2d(inputs, bottleneck_width, [1,1], stride=1, scope='conv_left_0')
 
            output = tf.concat([conv_right_0, conv_right_2], axis=3)
        return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)


  
@slim.add_arg_scope
def stem_block(inputs,
               num_init_channel=32,
               outputs_collections=None,
               scope=None):

    with tf.variable_scope(scope, 'stem_block', [inputs]) as sc:
        net = slim.conv2d(inputs, num_init_channel, [3,3], stride=2, scope='conv0')
                
        conv1_l0 = slim.conv2d(net, int(num_init_channel/2), [1, 1], stride=1, scope='conv1_l0')
        conv1_l1 = slim.conv2d(conv1_l0, num_init_channel, [3, 3], stride=2, scope='conv1_l1')
                
        maxpool1_r0 = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='maxpool1_r0')
                
        filter_concat = tf.concat([conv1_l1, maxpool1_r0], axis=-1)
                
        output = slim.conv2d(filter_concat, num_init_channel, 1, stride=1, scope='conv2')
        
    
    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)   
    
@slim.add_arg_scope
def transition_layer(inputs,is_avgpool=True, outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'transition', [inputs]) as sc:
        output_channel = inputs.get_shape().as_list()[-1]
        net = slim.conv2d(inputs, output_channel, [1,1], stride=1, scope='conv')
        if is_avgpool:
            output = slim.avg_pool2d(net, [2,2], stride=2, scope='avgpool')
        else:
            output = net
    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)  
    
@slim.add_arg_scope
def dense_block(inputs,
                 growth_rate,
                 bottleneck_width,
                 outputs_collections=None,
                 scope=None):
        with tf.variable_scope(scope, 'dense_block', [inputs]) as sc:
            growth_rate = int(growth_rate/2)
            inter_channel = int(growth_rate * bottleneck_width / 4) * 4
            
            # left channel
            conv_left_0 = slim.conv2d(inputs, inter_channel, [1,1], stride=1, scope='conv_left_0')
            conv_left_1 = slim.conv2d(conv_left_0, growth_rate, [3,3], stride=1, scope='conv_left_1')
            # right channel
            conv_right_0 = slim.conv2d(inputs, inter_channel, [1,1], stride=1, scope='conv_right_0')
            conv_right_1 = slim.conv2d(conv_right_0, growth_rate, [3,3], stride=1, scope='conv_right_1')
            conv_right_2 = slim.conv2d(conv_right_1, growth_rate, [3,3], stride=1, scope='conv_right_2')
                     
            output = tf.concat([inputs, conv_left_1, conv_right_2], axis=3)
        return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)


@slim.add_arg_scope
def stack_blocks_dense(net,
                       blocks,
                       outputs_collections=None):

    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
    
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    net = block.unit_fn(net, **dict(unit))
              
            net = transition_layer(net, is_avgpool = False if "block4" in sc.name else True)
            
    net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
    
    return net

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
  

def pelee_base(inputs,
                 num_classes=100,
                 dropout_keep_prob=0.999,
                 is_training=True,
                 min_depth=8,
                 depth_multiplier=1.0,
                 conv_defs=None,
                 prediction_fn=tf.contrib.layers.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='Pelee',
                 global_pool=True):

      input_shape = inputs.get_shape().as_list()
      if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                         len(input_shape))
      end_points = {}
  #with tf.variable_scope(scope, 'peee', [inputs]) as scope:
      end_points_collection = '_end_points' 
      with slim.arg_scope([slim.conv2d,slim.avg_pool2d,slim.max_pool2d,stem_block,dense_block],
                            outputs_collections = [end_points_collection]):
        with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=is_training):
            stem_block_output = stem_block(inputs, num_init_channel = 32)
        
        
            blocks = [Block("block1", dense_block, [{'growth_rate': 32,"bottleneck_width":1}]*3),
                      Block("block2", dense_block, [{'growth_rate': 32,"bottleneck_width":2}]*4),
                      Block("block3", dense_block, [{'growth_rate': 32,"bottleneck_width":4}]*8),
                      Block("block4", dense_block, [{'growth_rate': 32,"bottleneck_width":4}]*6)]
            
            print(blocks)
        
            net = stack_blocks_dense(stem_block_output,blocks)
            
            # Convert end_points_collection into a dictionary of end_points.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
#             with tf.variable_scope('Logits'):
#                 if global_pool:
#                     # Global average pooling.
#                     net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
#                     end_points['global_pool'] = net
#                 else:
#                     # Pooling with a fixed kernel size.
#                     kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
#                     net = slim.avg_pool2d(net, kernel_size, padding='VALID',
#                                         scope='AvgPool_1a')
#                 end_points['AvgPool_1a'] = net
#                 if not num_classes:
#                     return net, end_points
#                 # 1 x 1 x 1024
#                 net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
#                 logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
#                                      normalizer_fn=None, scope='Conv2d_1c_1x1')
#                 if spatial_squeeze:
#                     logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
#                 end_points['Logits'] = logits
        return end_points



def pelee(inputs,
                 num_classes=100,
                 dropout_keep_prob=0.999,
                 is_training=True,
                 min_depth=8,
                 depth_multiplier=1.0,
                 conv_defs=None,
                 prediction_fn=tf.contrib.layers.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='Pelee',
                 global_pool=True):

  input_shape = inputs.get_shape().as_list()
  if len(input_shape) != 4:
    raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                         len(input_shape))
  end_points = {}
  with tf.variable_scope(scope, 'Pelee', [inputs]) as scope:
      end_points_collection = scope.original_name_scope + '_end_points' 
      with slim.arg_scope([slim.conv2d,slim.avg_pool2d,slim.max_pool2d,stem_block,dense_block],
                            outputs_collections = [end_points_collection]):
        with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=is_training):
            stem_block_output = stem_block(inputs, num_init_channel = 32)
        
        
            blocks = [Block("block1", dense_block, [{'growth_rate': 32,"bottleneck_width":1}]*3),
                      Block("block2", dense_block, [{'growth_rate': 32,"bottleneck_width":2}]*4),
                      Block("block3", dense_block, [{'growth_rate': 32,"bottleneck_width":4}]*8),
                      Block("block4", dense_block, [{'growth_rate': 32,"bottleneck_width":4}]*6)]
            
            print(blocks)
        
            net = stack_blocks_dense(stem_block_output,blocks)
            
            # Convert end_points_collection into a dictionary of end_points.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            with tf.variable_scope('Logits'):
                if global_pool:
                    # Global average pooling.
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                    end_points['global_pool'] = net
                else:
                    # Pooling with a fixed kernel size.
                    kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
                    net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                        scope='AvgPool_1a')
                end_points['AvgPool_1a'] = net
                if not num_classes:
                    return net, end_points
                # 1 x 1 x 1024
                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                     normalizer_fn=None, scope='Conv2d_1c_1x1')
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
                end_points['Logits'] = logits
        return logits,end_points



def pelee_arg_scope(is_training=True,
                           weight_decay=0.00004,
                           stddev=0.09,
                           regularize_depthwise=False,
                           batch_norm_decay=0.9997,
                           batch_norm_epsilon=0.001):
  """Defines the default MobilenetV1 arg scope.

  Args:
    is_training: Whether or not we're training the model.
    weight_decay: The weight decay to use for regularizing the model.
    stddev: The standard deviation of the trunctated normal weight initializer.
    regularize_depthwise: Whether or not apply regularization on depthwise.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.

  Returns:
    An `arg_scope` to use for the mobilenet v1 model.
  """
  batch_norm_params = {
      'is_training': is_training,
      'center': True,
      'scale': True,
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
  }

  # Set weight_decay for weights in Conv and DepthSepConv layers.
  weights_init = tf.truncated_normal_initializer(stddev=stddev)
  regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  if regularize_depthwise:
    depthwise_regularizer = regularizer
  else:
    depthwise_regularizer = None
  with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                      weights_initializer=weights_init,
                      activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
        with slim.arg_scope([slim.separable_conv2d],
                            weights_regularizer=depthwise_regularizer) as sc:
          return sc

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
    
if __name__=='__main__':
    input_x = tf.Variable(tf.random_normal([1,224,224,3]))
    with slim.arg_scope(pelee_arg_scope()):
        output,end_points = pelee(input_x)
    
    print("Layers")
    for k, v in end_points.items():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))
      
    print("Parameters")
    for v in slim.get_model_variables():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))
    
    
    summary = tf.summary.FileWriter("logs",tf.get_default_graph())
    
    print(output.get_shape().as_list())
    
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        
        for i in range(10):
            start_time = time.time()
            _ = sess.run(output)
            duration = time.time() - start_time
            print ('%s: step %d, duration = %.3f' %(datetime.now(), i, duration))
    
    
    