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
import retinanet_model

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, \
    Normalizer
from oid_dataset import OidDataset

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

    parser.add_argument('--model', help='Path to model (.pt) file.')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)

    parser = parser.parse_args(args)

    if parser.dataset == 'coco':
        dataset_val = CocoDataset(parser.data_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))
    elif parser.dataset == 'openimages':
        dataset_val = OidDataset(parser.data_path, subset='validation',
                                 transform=transforms.Compose([Normalizer(), Resizer(min_side=400, max_side=600)]))
    elif parser.dataset == 'csv':
        dataset_val = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                 transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = retinanet_model.resnet18(num_classes=dataset_val.num_classes(), pretrained=False)
    elif parser.depth == 34:
        retinanet = retinanet_model.resnet34(num_classes=dataset_val.num_classes(), pretrained=False)
    elif parser.depth == 50:
        retinanet = retinanet_model.resnet50(num_classes=dataset_val.num_classes(), pretrained=False)
    elif parser.depth == 101:
        retinanet = retinanet_model.resnet101(num_classes=dataset_val.num_classes(), pretrained=False)
    elif parser.depth == 152:
        retinanet = retinanet_model.resnet152(num_classes=dataset_val.num_classes(), pretrained=False)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    checkpoint = torch.load(parser.model, map_location=lambda storage, loc: storage)
    weights = checkpoint['model']
    retinanet.load_state_dict(weights)

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet.eval()

    unnormalize = UnNormalizer()

    def draw_caption(image, box, caption):

        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    for idx, data in enumerate(dataloader_val):

        with torch.no_grad():
            st = time.time()
            if use_gpu:
                data_img = data['img'].cuda()
            else:
                data_img = data['img']
            scores, classification, transformed_anchors = retinanet(data_img.float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores > 0.5)
            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

            img[img < 0] = 0
            img[img > 255] = 255

            img = np.transpose(img, (1, 2, 0))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

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
