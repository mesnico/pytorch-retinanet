import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from models import RetinanetModel


def create_model(num_classes, parser):
    if parser.net == 'retinanet':
        model = RetinanetModel(num_classes, parser.depth, pretrained=True)
    elif parser.net == 'fasterrcnn':
        # load a model pre-trained pre-trained on COCO
        if parser.depth == 50:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        elif parser.depth == 101:
            model = torchvision.models.detection.fasterrcnn_resnet101_fpn(pretrained=True)
        else:
            raise ValueError('Unsupported model depth, must be one of 50, 101')
        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        # num_classes += 1  # add background
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        raise ValueError('Unsupported net!')

    return model
