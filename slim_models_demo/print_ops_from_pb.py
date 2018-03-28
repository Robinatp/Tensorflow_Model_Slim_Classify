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


checkpoints_dir = '../tmp/checkpoints'
# print_ops(os.path.join(checkpoints_dir, 'vgg_16_freeze_graph.pb'))
print_ops("../tmp/checkpoints/with_placeholder/frozen_graph.pb")
