# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import os
import argparse


# print all op names
def print_ops(pb_path,output_layer):
    with tf.gfile.FastGFile(os.path.join(pb_path), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
        
    with tf.Session() as sess:
        
        ops = sess.graph.get_operations()
        for op in ops:
            print(op.name)

        writer =tf.summary.FileWriter("log_print_ops/",graph = sess.graph)
        writer.close()
        
        graph = tf.get_default_graph()
        input = graph.get_tensor_by_name(output_layer)
        print(input)



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
 
  parser.add_argument("--pb_path", default='tmp/frozen_graph.pb',help="name of pb_path")
  parser.add_argument("--output_layer",default='MobilenetV1/Predictions/Reshape_1:0', help="name of output layer")
  args = parser.parse_args()
print_ops(args.pb_path,args.output_layer)
