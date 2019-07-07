import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(num_classes, parser):
    if parser.net == 'retinanet':
        if parser.depth == 18:
            model = retinanet_model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
        elif parser.depth == 34:
            model = retinanet_model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
        elif parser.depth == 50:
            model = retinanet_model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
        elif parser.depth == 101:
            model = retinanet_model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
        elif parser.depth == 152:
            model = retinanet_model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    elif parser.net == 'fastercnn':
        # load a model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        num_classes += 1  # add background
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        raise ValueError('Unsupported net!')

    return model