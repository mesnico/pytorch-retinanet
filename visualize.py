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
from datasets.oid_dataset import OidDataset
from create_model import create_model

# assert torch.__version__.split('.')[1] == '4'

use_gpu = False
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

    parser.add_argument('--model', help='Path to model (.pt) file.')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)

    parser = parser.parse_args(args)

    if parser.dataset == 'coco':
        raise NotImplementedError()
        dataset_val = CocoDataset(parser.data_path, set_name='val2017',
                                  transform=Compose([Normalizer(), Resizer()]))
    elif parser.dataset == 'openimages':
        dataset_val = OidDataset(parser.data_path, subset='validation',
                                 transform=Compose([ToTensor()]))
    elif parser.dataset == 'csv':
        raise NotImplementedError()
        dataset_val = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                 transform=Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collate_fn, batch_sampler=sampler_val)

    # Create the model
    model = create_model(dataset_val.num_classes(), parser)

    checkpoint = torch.load(parser.model, map_location=lambda storage, loc: storage)
    weights = checkpoint['model']
    model.load_state_dict(weights)

    if use_gpu:
        model = model.cuda()

    model.eval()

    def draw_caption(image, box, caption):

        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

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
            transformed_anchors = output['boxes']
            # from here, interface to the code already written in the original repo

            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores > 0.5)
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

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                print(label_name)
            '''

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                draw_caption(img, (x1, y1, x2, y2), label_name)

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                print(label_name)

            cv2.imshow('img', img)
            cv2.waitKey(0)


if __name__ == '__main__':
    main()
