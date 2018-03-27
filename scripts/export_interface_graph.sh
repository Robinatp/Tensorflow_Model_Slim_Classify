SLIM_NAME=mobilenet_v1
MODEL_FOLDER=./tmp/
DATASET_DIR=/workspace/zhangbin/dataset_robin/flowers
echo "Freezing graph to ${MODEL_FOLDER}/unfrozen_graph.pb"
python  export_inference_graph.py \
  --model_name=${SLIM_NAME} \
  --image_size=224 \
  --dataset_name=flowers \
  --dataset_dir=${DATASET_DIR} \
  --logtostderr \
  --output_file=${MODEL_FOLDER}/unfrozen_graph.pb 
  
  
echo "*******"
echo "Freezing graph to ${MODEL_FOLDER}/frozen_graph.pb"
echo "*******"
CHECKPOINT=./tmp/flowers-models/mobilenet_v1/all/model.ckpt-5032
OUTPUT_NODE_NAMES=MobilenetV1/Predictions/Reshape_1
python tools/freeze_graph.py \
  --input_graph=${MODEL_FOLDER}/unfrozen_graph.pb \
  --input_checkpoint=${CHECKPOINT} \
  --input_binary=true \
  --output_graph=${MODEL_FOLDER}/frozen_graph.pb \
  --output_node_names=${OUTPUT_NODE_NAMES}

python tools/optimize_for_inference.py \
--input=${MODEL_FOLDER}/frozen_graph.pb \
--output=${MODEL_FOLDER}/optimized_graph.pb \
--frozen_graph=True \
--input_names=input \
--output_names=${OUTPUT_NODE_NAMES}


python tools/quantization/quantize_graph.py \
--input=${MODEL_FOLDER}/frozen_graph.pb \
--output_node_names=${OUTPUT_NODE_NAMES} \
--print_nodes \
--output=${MODEL_FOLDER}/quantized_graph.pb \
--mode=eightbit \
--logtostderr