# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import os


# print all op names
def print_ops(pb_path):
    with tf.gfile.FastGFile(os.path.join(pb_path), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
        
    with tf.Session() as sess:
        
        ops = sess.graph.get_operations()
        for op in ops:
            print(op.name)

        writer =tf.summary.FileWriter("log_print_ops/",sess.graph)
        writer.close()

print_ops('model/freeze_model.pb')
