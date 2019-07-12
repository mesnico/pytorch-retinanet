import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse
import tqdm

import sys
import cv2
from models.retinanet import model

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models

from transforms import Compose, RandomHorizontalFlip, ToTensor
from dataloader import collate_fn, AspectRatioBasedSampler, \
    UnNormalizer, Normalizer
from datasets import OidDataset
from create_model import create_model

# assert torch.__version__.split('.')[1] == '4'

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

    parser.add_argument('--model', help='Path to model (.pt) file.')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)

    parser = parser.parse_args(args)

    if parser.dataset == 'coco':
        raise NotImplementedError()
        dataset = CocoDataset(parser.data_path, set_name='val2017',
                                  transform=Compose([Normalizer(), Resizer()]))
    elif parser.dataset == 'openimages':
        dataset = OidDataset(parser.data_path, subset=parser.set,
                                 transform=Compose([ToTensor()]))
    elif parser.dataset == 'csv':
        raise NotImplementedError()
        dataset = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                 transform=Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    # sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader = DataLoader(dataset, num_workers=1, collate_fn=collate_fn, shuffle=False)

    # Create the model
    model = create_model(dataset.num_classes(), parser)

    checkpoint = torch.load(parser.model, map_location=lambda storage, loc: storage)
    weights = checkpoint['model']
    model.load_state_dict(weights)

    if use_gpu:
        model = model.cuda()

    model.eval()

    all_detections = []
    det_output_path = os.path.split(parser.model)[0]

    for idx, data in enumerate(tqdm.tqdm(dataloader)):
        with torch.no_grad():
            st = time.time()

            images, targets = data

            # targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
            if use_gpu:
                input_images = list(image.cuda().float() for image in images)
            else:
                input_images = list(image.float() for image in images)

            outputs = model(input_images)
            outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

            output = outputs[0]  # take the only batch
            scores = output['scores']
            classification = output['labels']
            transformed_anchors = output['boxes']
            # from here, interface to the code already written in the original repo

            # TODO: 0.5 should be a parameter in a configuration file.. that hopefully should be created and handled..
            det_idxs = np.where(scores > 0.5)

            bboxes = transformed_anchors[det_idxs[0][det_idxs], :].cpu().numpy()
            labels = classification[det_idxs[0][det_idxs]].cpu().numpy()
            scores = scores[det_idxs[0][det_idxs]].cpu().numpy()

            packed_detections = [idx, bboxes, labels, scores]
            all_detections.append(packed_detections)

            # if idx == 20:
            #    break

    print('Evaluating...')
    # TODO: add identification parameter to evaluate so that detections from different checkpoints are not overwritten
    dataset.evaluate(all_detections, det_output_path, file_identifier=parser.set)
    print('DONE!')


if __name__ == '__main__':
    main()
