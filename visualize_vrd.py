import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2
from models.retinanet import model

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models

from transforms import Compose, RandomHorizontalFlip, ToTensor
from dataloader import collate_fn, AspectRatioBasedSampler, UnNormalizer, Normalizer
from datasets.oid_dataset import OidDataset, OidDatasetVRD
from models.create_model import create_detection_model
from models.vrd import VRD

# assert torch.__version__.split('.')[1] == '4'
thres = 0.4
rel_thresh = 0.1
attr_thresh = 0.1
max_objects = 80

use_gpu = True

if not use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--data_path', help='Path to COCO directory')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--net', help='Network to use', default='fasterrcnn')
    parser.add_argument('--set', help='Set on which evaluation will be performed', default='validation')

    parser.add_argument('--model_rel', help='Path to model (.pt) file for relationships.', default=None)
    parser.add_argument('--model_attr', help='Path to model (.pt) file for attributes.', default=None)
    parser.add_argument('--model_detector', help='Path to model (.pt) file for the detector.')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)

    parser = parser.parse_args(args)

    if parser.dataset == 'openimages':
        dataset_val = OidDatasetVRD(parser.data_path, subset=parser.set,
                                    transform=Compose([ToTensor()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    #sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collate_fn, batch_size=1, shuffle=True)

    # Create the model
    detector = create_detection_model(dataset_val.num_classes(), parser, box_score_thresh=thres)
    model = VRD(detector, dataset=dataset_val, train_relationships=parser.model_rel is not None, 
                train_attributes=parser.model_attr is not None, max_objects=max_objects)

    # Load the detector
    checkpoint = torch.load(parser.model_detector, map_location=lambda storage, loc: storage)
    weights = checkpoint['model']
    weights = {k.replace('module.', ''): v for k, v in weights.items()}
    model.detector.load_state_dict(weights)
    print('Detector correctly loaded!')

    # Load the attributes, if needed
    if parser.model_rel:
        checkpoint = torch.load(parser.model_rel, map_location=lambda storage, loc: storage)
        weights = checkpoint['model_rel']
        weights = {k.replace('module.', ''): v for k, v in weights.items()}
        model.relationships_net.load_state_dict(weights)
        print('Relationships correctly loaded!')

    if parser.model_attr:
        checkpoint = torch.load(parser.model_attr, map_location=lambda storage, loc: storage)
        weights = checkpoint['model_attr']
        weights = {k.replace('module.', ''): v for k, v in weights.items()}
        model.attributes_net.load_state_dict(weights)
        print('Attributes correctly loaded!')

    if use_gpu:
        model = model.cuda()

    model.eval()

    def draw_object_bb(image, box, caption):
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), color=(0, 0, 255), thickness=2)

    def draw_relationship(image, subj, obj, rel_name):
        cv2.arrowedLine(image, (subj[0], subj[1]), (obj[0], obj[1]), (255, 0, 0), 2, tipLength=0.02)
        cv2.putText(image, rel_name, ((subj[0] + obj[0]) / 2, (subj[1] + obj[1]) / 2), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    for idx, data in enumerate(dataloader_val):
        with torch.no_grad():
            st = time.time()

            images, targets = data

            # targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
            if use_gpu:
                input_images = list(image.cuda().float() for image in images)
            else:
                input_images = list(image.float() for image in images)
            # TODO: adapt retinanet output to the one by torchvision 0.3
            # scores, classification, transformed_anchors = model(data_img.float())
            outputs = model(input_images)
            outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

            output = outputs[0]  # take the only batch
            scores = output['scores']
            classification = output['labels']
            boxes = output['boxes']
            if parser.model_rel:
                relationships = output['relationships']
                rel_scores = output['relationships_scores']
            if parser.model_attr:
                attributes = output['attributes']
                attr_scores = output['attributes_scores']
            # from here, interface to the code already written in the original repo

            print('Elapsed time: {}'.format(time.time() - st))
            img = np.array(255 * images[0]).copy()

            img[img < 0] = 0
            img[img > 255] = 255

            img = np.transpose(img, (1, 2, 0))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            '''
            # Visualize ground truth bounding boxes
            for bbox, label in zip(targets[0]['boxes'], targets[0]['labels']):
                # bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = dataset_val.labels[int(label)]
                draw_caption(img, (x1, y1, x2, y2), label_name)

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)
                print('GT: '+label_name)
            '''
            if len(boxes) != 0:

                # Draw objects
                for j in range(attributes.shape[0]):
                    bbox = boxes[j, :4].int()
                    attr = attributes[j, 0].item() if parser.model_attr is not None and attr_scores[j, 0] > attr_thresh else 0      # TODO: only the top rank attribute is considered, generalize better!
                    label_name = dataset_val.labels[int(classification[j])]
                    attr_name = ': ' + dataset_val.attr_id_to_labels[attr] if attr != 0 else ''
                    draw_object_bb(img, bbox, label_name + attr_name)
                    print('Detection: '+label_name)

                # Draw relationships
                if parser.model_rel:
                    for s_ind in range(relationships.shape[0]):
                        for o_ind in range(relationships.shape[1]):
                            subj = boxes[s_ind, :4].int()
                            obj = boxes[o_ind, :4].int()
                            rel = relationships[s_ind, o_ind].item() if rel_scores[s_ind, o_ind] > rel_thresh else 0
                            if rel != 0:
                                rel_name = dataset_val.rel_id_to_labels[rel]
                                draw_relationship(img, subj, obj, rel_name)

            cv2.imshow('img', img)
            cv2.waitKey(0)


if __name__ == '__main__':
    main()
