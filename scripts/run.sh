protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
echo $PYTHONPATH

PATH_TO_YOUR_PIPELINE_CONFIG=`pwd`/object_detection/01_pet_dataset/model/mask_rcnn_inception_v2_coco_2018_01_28/mask_rcnn_inception_v2_coco.config
PATH_TO_TRAIN_DIR=`pwd`/object_detection/01_pet_dataset/model/mask_rcnn_inception_v2_coco_2018_01_28/train
PATH_TO_EVAL_DIR=`pwd`/object_detection/01_pet_dataset/model/mask_rcnn_inception_v2_coco_2018_01_28/eval

echo $PATH_TO_YOUR_PIPELINE_CONFIG
##test the env
python object_detection/builders/model_builder_test.py


echo "input command:train or eval or export:"
read a
echo "input is $a"


##create the tfrecord files
#python object_detection/create_pet_tf_record.py \
#    --label_map_path=`pwd`/object_detection/pets_tf_tutorials/data/pet_label_map.pbtxt \
#    --data_dir=`pwd`/object_detection/pets_tf_tutorials/data/dataset/ \
#    --output_dir=`pwd`/object_detection/pets_tf_tutorials/data/

if [ $a = train ] ; then

## From the tensorflow/models/research/ directory
#python object_detection/train.py \
#    --logtostderr \
#    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
#    --train_dir=${PATH_TO_TRAIN_DIR}

python object_detection/train.py \
    --logtostderr \
   --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --train_dir=${PATH_TO_TRAIN_DIR}
fi


if [ $a = eval ] ; then
## From the tensorflow/models/research/ directory
#python object_detection/eval.py \
#    --logtostderr \
#    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
#    --checkpoint_dir=${PATH_TO_TRAIN_DIR} \
#    --eval_dir=${PATH_TO_EVAL_DIR}


python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --checkpoint_dir=${PATH_TO_TRAIN_DIR} \
    --eval_dir=${PATH_TO_EVAL_DIR}
fi


if [ $a = export ] ; then
## From tensorflow/models/research/
#python object_detection/export_inference_graph.py \
#    --input_type image_tensor \
#    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
#    --trained_checkpoint_prefix ${TRAIN_PATH} \
#    --output_directory output_inference_graph.pb

python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --trained_checkpoint_prefix=`pwd`/object_detection/01_pet_dataset/model/mask_rcnn_inception_v2_coco_2018_01_28/train/model.ckpt-83540 \
    --output_directory=`pwd`/object_detection/01_pet_dataset/model/mask_rcnn_inception_v2_coco_2018_01_28/fine_tuned_model/model-83540
	
	
#python object_detection/export_inference_graph.py \
#    --input_type image_tensor \
#    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
#    --trained_checkpoint_prefix=`pwd`/object_detection/02_hands_tf_tutorials/Egohands_models/ssd_inception_v2_coco_2017_11_17/train/model.ckpt-45322 \
#    --output_directory=`pwd`/object_detection/02_hands_tf_tutorials/Egohands_models/ssd_inception_v2_coco_2017_11_17/fine_tuned_model/model-45322
fi
