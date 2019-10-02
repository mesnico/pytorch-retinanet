import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse
import tqdm
import shelve

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
thres = 0.2
rel_thresh = 0.1
attr_thresh = 0.1
max_objects = 200

use_gpu = True

if not use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    global thres
    global rel_thresh
    global attr_thresh

    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--data_path', help='Path to COCO directory')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--net', help='Network to use', default='fasterrcnn')
    parser.add_argument('--set', help='Set on which evaluation will be performed', default='validation')
    parser.add_argument('--store_detections', action='store_true', default=False,
                        help='Cache all detections with very low threshold in order to enable filtering after extraction')
    parser.add_argument('--load_detections', action='store_true', default=False,
                        help='Load cached detections')

    parser.add_argument('--model_rel', help='Path to model (.pt) file for relationships.', default=None)
    parser.add_argument('--model_attr', help='Path to model (.pt) file for attributes.', default=None)
    parser.add_argument('--model_detector', help='Path to model (.pt) file for the detector.')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)

    parser = parser.parse_args(args)

    assert parser.model_rel is not None and parser.model_attr is not None and parser.model_detector is not None, \
           'Models snapshots have to be specified!'
    assert not (parser.load_detections and parser.store_detections)

    det_output_path = os.path.split(parser.model_rel)[0]

    if parser.dataset == 'openimages':
        dataset_val = OidDatasetVRD(parser.data_path, subset=parser.set,
                                    transform=Compose([ToTensor()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    #sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collate_fn, batch_size=1)

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

    all_detections = []
    if parser.load_detections or parser.store_detections:
        print('Opening detections database file...')
        flag = 'r' if parser.load_detections else 'c'
        loaded_detections = shelve.open(os.path.join(det_output_path, 'cached_detections_detthr{}.db'.format(thres)), flag=flag)

    for idx, data in enumerate(tqdm.tqdm(dataloader_val)):
        if parser.load_detections:
            loaded_det = loaded_detections[str(idx)]
            scores = loaded_det[0]
            classification = loaded_det[1]
            boxes = loaded_det[2]
            relationships = loaded_det[3]
            rel_scores = loaded_det[4]
            attributes = loaded_det[5]
            attr_scores = loaded_det[6]
        else:
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
            relationships = output['relationships']
            rel_scores = output['relationships_scores']
            attributes = output['attributes']
            attr_scores = output['attributes_scores']

        if parser.store_detections:
            loaded_detections[str(idx)] = [scores, classification, boxes, relationships, rel_scores, attributes, attr_scores]
        else:
            '''if parser.load_detections:
                pdb.set_trace()
                # filter objects, relationships and attributes
                filtered_idxs = np.where(scores > thres)[0]
                scores = scores[filtered_idxs]
                classification = classification[filtered_idxs]
                boxes = boxes[filtered_idxs]
                relationships = relationships[np.ix_(filtered_idxs, filtered_idxs)]
                rel_scores = rel_scores[np.ix_(filtered_idxs, filtered_idxs)]
                attributes = attributes[filtered_idxs]
                attr_scores = attr_scores[filtered_idxs]
            '''
            subj_boxes_out = []
            subj_labels_out = []
            obj_boxes_out = []
            obj_labels_out = []
            rel_labels_out = []
            rel_scores_out = []

            if len(boxes) != 0:
                # num_objects = min(boxes.shape[0], max_objects)

                # Collect objects and attributes
                for j in range(attributes.shape[0]):
                    bbox = boxes[j, :4]
                    attr = attributes[j, 0].item() if parser.model_attr is not None and attr_scores[j, 0] > attr_thresh else 0      # TODO: only the top rank attribute is considered, generalize better!
                    # We add an 'is' relation. 'is' relation is mapped to relation index of -1.
                    if attr != 0:
                        subj_boxes_out.append(bbox)
                        obj_boxes_out.append(bbox)
                        rel_labels_out.append(-1)
                        rel_scores_out.append(attr_scores[j, 0])
                        subj_labels_out.append(int(classification[j]))
                        obj_labels_out.append(attr)
                # Collect relationships
                for s_ind in range(relationships.shape[0]):
                    for o_ind in range(relationships.shape[1]):
                        subj = boxes[s_ind, :4]
                        obj = boxes[o_ind, :4]
                        rel = relationships[s_ind, o_ind].item() if rel_scores[s_ind, o_ind] > rel_thresh else 0
                        if rel != 0:
                            subj_boxes_out.append(subj)
                            obj_boxes_out.append(obj)
                            rel_labels_out.append(rel)
                            rel_scores_out.append(rel_scores[s_ind, o_ind])
                            subj_labels_out.append(int(classification[s_ind]))
                            obj_labels_out.append(int(classification[o_ind]))

            all_detections.append([idx, subj_boxes_out, subj_labels_out, obj_boxes_out, obj_labels_out, rel_labels_out, rel_scores_out])
            # if idx == 400:
            #    break

    if not parser.store_detections:
        print('Evaluating...')
        # TODO: add identification parameter to evaluate so that detections from different checkpoints are not overwritten
        dataset_val.evaluate(all_detections, det_output_path, file_identifier='{}_relthr{}_attrthr{}_detthr{}'.format(parser.set, rel_thresh, attr_thresh, thres))
        print('DONE!')


if __name__ == '__main__':
    main()
