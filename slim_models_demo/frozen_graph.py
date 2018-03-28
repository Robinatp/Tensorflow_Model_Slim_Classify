import tensorflow as tf
from tensorflow.python.framework import graph_util
import os
import PIL.Image as Image
import  numpy as np
def freeze_graph(model_dir, output_node_names):
    """
    freeze the saved checkpoints/graph to *.pb
    """
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    output_graph = os.path.join(model_dir, "frozen_graph.pb")
    
    saver = tf.train.import_meta_graph(input_checkpoint + ".meta", 
                                       clear_devices=True)

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        output_graph_def = graph_util.convert_variables_to_constants(sess,
                                                                     input_graph_def,
                                                                     output_node_names.split(","))

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph" % (len(output_graph_def.node)))
        
        
def load_graph(frozen_graph_filename):
    """
    Loads Frozen graph
    """
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph

#mobilenet
# freeze_graph("tf_files/mobilenet/", output_node_names="final_result")      
# graph = load_graph("tf_files/mobilenet/frozen_graph.pb")
# 
# for op in graph.get_operations():
#     print(op.name)
#    
# 
# input_x = graph.get_tensor_by_name("import/input:0")
# print(input_x)
# out = graph.get_tensor_by_name("import/final_result:0")    
# print(out)
# 
# input_operation = graph.get_operation_by_name('import/input')
# print(input_operation.outputs[0])
# output_operation = graph.get_operation_by_name('import/final_result')
# print(output_operation.outputs[0])

#inception
freeze_graph("../tmp/checkpoints/with_placeholder/", output_node_names="vgg_16/fc8/squeezed")      
graph = load_graph("../tmp/checkpoints/with_placeholder/frozen_graph.pb")

for op in graph.get_operations():
    print(op.name)
   

input_x = graph.get_tensor_by_name("import/input:0")
print(input_x)
out = graph.get_tensor_by_name("import/vgg_16/fc8/squeezed:0")    
print(out)

input_operation = graph.get_operation_by_name('import/input')
print(input_operation.outputs[0])
output_operation = graph.get_operation_by_name('import/vgg_16/fc8/squeezed')
print(output_operation.outputs[0])







  