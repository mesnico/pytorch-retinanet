#!/bin/bash

BOUNDING_BOXES=/media/nicola/SSD/Datasets/OpenImages/metadata/validation-annotations-bbox
IMAGE_LABELS=/media/nicola/SSD/Datasets/OpenImages/metadata/validation-annotations-human-imagelabels-boxable
LABEL_MAP=/media/nicola/SSD/Datasets/OpenImages/metadata/class_dict_labelmap_600.pbtxt

INPUT_PREDICTIONS=/media/nicola/Data/Workspace/VRD-OpenImages/pytorch-retinanet/outputs/experiment_fasterrcnn_openimages_resnet50_20190711214520/detections_validation_IoU0.4.csv
OUTPUT_METRICS=/media/nicola/Data/Workspace/VRD-OpenImages/pytorch-retinanet/outputs/experiment_fasterrcnn_openimages_resnet50_20190711214520/final_oid_detections

cd /media/nicola/Data/Workspace/VRD-OpenImages/tf_evaluation/research
python object_detection/metrics/oid_challenge_evaluation.py \
    --input_annotations_boxes=${BOUNDING_BOXES}_expanded.csv \
    --input_annotations_labels=${IMAGE_LABELS}_expanded.csv \
    --input_class_labelmap=${LABEL_MAP} \
    --input_predictions=${INPUT_PREDICTIONS} \
    --output_metrics=${OUTPUT_METRICS} \
