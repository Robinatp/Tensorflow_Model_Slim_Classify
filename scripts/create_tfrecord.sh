python ../../dataset_tools/create_pet_tf_record.py \
        --data_dir=/workspace/zhangbin/master/tensorflow_models/models/research/object_detection/01_pet_dataset/dataset \
        --output_dir=/workspace/zhangbin/master/tensorflow_models/models/research/object_detection/01_pet_dataset/dataset \
        --label_map_path=/workspace/zhangbin/master/tensorflow_models/models/research/object_detection/01_pet_dataset/dataset/pet_label_map.pbtxt \
        --faces_only=False
