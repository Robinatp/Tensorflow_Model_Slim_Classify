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
def stem_block(inputs,
               num_init_channel=64,
               outputs_collections=None,
               scope=None):

    with tf.variable_scope(scope, 'stem_block', [inputs]) as sc:
        net = slim.conv2d(inputs, num_init_channel, [3,3], stride=2, scope='conv0')
        net = slim.conv2d(net, num_init_channel, [3,3], stride=1, scope='conv1')
        
        net = slim.conv2d(net, num_init_channel*2, [1,1], stride=1, scope='conv2')
        
        output = slim.avg_pool2d(net, [2,2], stride=2, scope='maxpool0')
        
    
    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)   
    
@slim.add_arg_scope
def transition_layer(inputs, is_avgpool=True, outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'transition', [inputs]) as sc:
        output_channel = inputs.get_shape().as_list()[-1]
        
        if is_avgpool:
            net = slim.conv2d(inputs, output_channel, [1,1], stride=1, scope='conv')
            output = slim.max_pool2d(net, [2,2], stride=2, scope='maxpool')
        else:
            output = slim.conv2d(inputs, output_channel, [1,1], stride=1, scope='conv')
    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)
    
    

@slim.add_arg_scope
def dense_block(inputs,
                 growth_rate,
                 outputs_collections=None,
                 scope=None):
        with tf.variable_scope(scope, 'dense_block', [inputs]) as sc:
            residual = slim.conv2d(inputs, growth_rate, [1,1], stride=1, scope='conv_1x1')
            
            residual = slim.conv2d(residual, growth_rate, [3,3], stride=1, scope='conv_3x3')
            
            output = tf.concat([residual, inputs], axis=3)
            
        
        return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)


@slim.add_arg_scope
def dense_block_b(inputs,
                 growth_rate,
                 outputs_collections=None,
                 scope=None):
        with tf.variable_scope(scope, 'dense_block', [inputs]) as sc:
            residual = slim.conv2d(inputs, growth_rate, [1,1], stride=1, scope='conv_1x1')
            
            residual = slim.separable_conv2d(residual, None, [3, 3], stride =1,scope="dwconv_3X3")
            
            output = tf.concat([residual, inputs], axis=3)
            
        
        return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)
        
        
@slim.add_arg_scope
def dense_block_a(inputs,
                 growth_rate,
                 expand_ratio=0.2,
                 outputs_collections=None,
                 scope=None):
        with tf.variable_scope(scope, 'dense_block', [inputs]) as sc:
            input_channel = inputs.get_shape().as_list()[-1]
            
            residual = slim.conv2d(inputs, int(input_channel*expand_ratio), [1,1], stride=1, scope='conv_1x1_in')
            
            residual = slim.separable_conv2d(residual, None, [3, 3], stride =1,scope="dwconv_3X3")
            
            residual = slim.conv2d(residual, growth_rate, [1,1], stride=1, scope='conv_1x1_out')
            
            output = tf.concat([residual, inputs], axis=3)
            
        
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
              
#             net = transition_layer(net, is_avgpool = False if "block4" in sc.name else True)
            
    net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
    
    return net


def dsod(inputs,
                 num_classes=100,
                 dropout_keep_prob=0.999,
                 is_training=True,
                 min_depth=8,
                 depth_multiplier=1.0,
                 conv_defs=None,
                 prediction_fn=tf.contrib.layers.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='dsod',
                 global_pool=True):

  input_shape = inputs.get_shape().as_list()
  if len(input_shape) != 4:
    raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                         len(input_shape))
  end_points = {}
  with tf.variable_scope(scope, 'dsod', [inputs]) as scope:
      end_points_collection = scope.original_name_scope + '_end_points' 
      with slim.arg_scope([slim.conv2d,slim.avg_pool2d,slim.max_pool2d,stem_block,dense_block,transition_layer],
                            outputs_collections = [end_points_collection]):
        with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=is_training):
            stem_block_output = stem_block(inputs, num_init_channel = 64)
        
        
            blocks = [Block("stage1", dense_block, [{'growth_rate': 48}]*6),
                      Block("transition_layer1", transition_layer, [{"is_avgpool":True}]),
                      Block("stage2", dense_block, [{'growth_rate': 48}]*8),
                      Block("transition_layer2", transition_layer, [{"is_avgpool":True}]),
                      Block("stage3", dense_block, [{'growth_rate': 48}]*8),
                      Block("transition_layer3", transition_layer, [{"is_avgpool":False}]),
                      Block("stage4", dense_block, [{'growth_rate': 48}]*8),
                      Block("transition_layer4", transition_layer, [{"is_avgpool":False}])
                      ]
            
            print(blocks)
        
            net = stack_blocks_dense(stem_block_output,blocks)
            
            # Convert end_points_collection into a dictionary of end_points.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            with tf.variable_scope('Logits'):
                if global_pool:
                    # Global average pooling.
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                    end_points['global_pool'] = net
#                 else:
#                     # Pooling with a fixed kernel size.
#                     kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
#                     net = slim.avg_pool2d(net, kernel_size, padding='VALID',
#                                         scope='AvgPool_1a')
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


def dsod_arg_scope(
    is_training=True,
    weight_decay=0.00004,
    stddev=0.09,
    regularize_depthwise=False,
    batch_norm_decay=0.9997,
    batch_norm_epsilon=0.001,
    batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):
  """Defines the default MobilenetV1 arg scope.

  Args:
    is_training: Whether or not we're training the model. If this is set to
      None, the parameter is not added to the batch_norm arg_scope.
    weight_decay: The weight decay to use for regularizing the model.
    stddev: The standard deviation of the trunctated normal weight initializer.
    regularize_depthwise: Whether or not apply regularization on depthwise.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.
    batch_norm_updates_collections: Collection for the update ops for
      batch norm.

  Returns:
    An `arg_scope` to use for the mobilenet v1 model.
  """
  batch_norm_params = {
      'center': True,
      'scale': True,
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'updates_collections': batch_norm_updates_collections,
  }
  if is_training is not None:
    batch_norm_params['is_training'] = is_training

  # Set weight_decay for weights in Conv and DepthSepConv layers.
  weights_init = tf.truncated_normal_initializer(stddev=stddev)
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


def dsod_base(inputs,
                 num_classes=100,
                 dropout_keep_prob=0.999,
                 is_training=True,
                 min_depth=8,
                 depth_multiplier=1.0,
                 conv_defs=None,
                 prediction_fn=tf.contrib.layers.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='dsod',
                 global_pool=True):

  input_shape = inputs.get_shape().as_list()
  if len(input_shape) != 4:
    raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                         len(input_shape))
  end_points = {}
  
  end_points_collection = scope.original_name_scope + '_end_points' 
  with slim.arg_scope([slim.conv2d,slim.avg_pool2d,slim.max_pool2d,stem_block,dense_block,transition_layer],
                        outputs_collections = [end_points_collection]):
    with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=is_training):
        stem_block_output = stem_block(inputs, num_init_channel = 64)
    
    
        blocks = [Block("stage1", dense_block, [{'growth_rate': 48}]*6),
                  Block("transition_layer1", transition_layer, [{"is_avgpool":True}]),
                  Block("stage2", dense_block, [{'growth_rate': 48}]*8),
                  Block("transition_layer2", transition_layer, [{"is_avgpool":True}]),
                  Block("stage3", dense_block, [{'growth_rate': 48}]*8),
                  Block("transition_layer3", transition_layer, [{"is_avgpool":False}]),
                  Block("stage4", dense_block, [{'growth_rate': 48}]*8),
                  Block("transition_layer4", transition_layer, [{"is_avgpool":False}])
                  ]
        
        print(blocks)
    
        net = stack_blocks_dense(stem_block_output,blocks)
        
        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        
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

   
if __name__=='__main__':
    input_x = tf.Variable(tf.random_normal([1,300,300,3]))
    with slim.arg_scope(dsod_arg_scope(is_training = False)):
        logits,end_points = dsod(input_x, is_training=False)
    
    
    print("Layers")
    for k, v in end_points.items():
        print('k={}, name = {}, shape = {}'.format(k,v.name, v.get_shape()))
       
    print("Parameters")
    for v in slim.get_model_variables():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))
     
     
    summary = tf.summary.FileWriter("logs",tf.get_default_graph())
    
#     print(output.get_shape().as_list())
    
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(10):
            start_time = time.time()
            _ = sess.run(logits)
            duration = time.time() - start_time
            print ('%s: step %d, duration = %.3f' %(datetime.now(), i, duration))
        