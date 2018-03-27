#-*- coding:utf-8 -*-
import argparse 
import numpy as np
import tensorflow as tf
import sys
from tensorflow.python.platform import gfile

def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="", 
            op_dict=None, 
            producer_op_list=None
        )   

        writer = tf.summary.FileWriter("./logs_inception_from_freeze_graph", graph=graph)
        writer.close() 
        
    return graph

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label
  
def main(_):
    #load the freeze graph and return graph
    graph = load_graph(FLAGS.frozen_model_filename)

    # We can list operations
    #op.values() gives you a list of tensors it produces
    #op.name gives you the name
    for op in graph.get_operations():
#         print(op.name,op.values())
        print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions
    #操作有:prefix/Placeholder/inputs_placeholder
    #操作有:prefix/Accuracy/predictions
    #为了预测,我们需要找到我们需要feed的tensor,那么就需要该tensor的名字
    #注意prefix/Placeholder/inputs_placeholder仅仅是操作的名字,prefix/Placeholder/inputs_placeholder:0才是tensor的名字
    x = graph.get_tensor_by_name(FLAGS.input_tensor_name)
    y = graph.get_tensor_by_name(FLAGS.output_tensor_name)
        
    with tf.Session(graph=graph) as sess:
        image_data = gfile.FastGFile(FLAGS.image_dir, 'rb').read()
        pre = sess.run(y, feed_dict={x:image_data})
        print(pre)
    
        results = np.squeeze(pre)
        classes =load_labels(FLAGS.output_labels)
        top_k = results.argsort()[-5:][::-1]
        for i in top_k:
            print(classes[i], results[i])
        print ("finish")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--frozen_model_filename", 
        default="model/freeze_model.pb", 
        type=str, 
        help="Frozen model file to import"
    )
    
    parser.add_argument(
        '--output_labels',
        type=str,
        default='tmp/retrained_labels.txt',
        help='Where to load the trained graph\'s labels.'
    )
    
    parser.add_argument(
        '--image_dir',
        type=str,
        default='flower_data/sunflowers/1022552002_2b93faf9e7_n.jpg',
        help='Path to folders of labeled images.'
    )
    
    parser.add_argument(
        '--input_tensor_name',
        type=str,
        default='import/DecodeJpeg/contents:0',
        help="""\
          The name of the output classification layer in the retrained graph.\
          """
    )
    
    parser.add_argument(
        '--output_tensor_name',
        type=str,
        default='final_training_ops/Softmax:0',
        help="""\
          The name of the output classification layer in the retrained graph.\
          """
    )
    
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  