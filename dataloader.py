from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler
import torchvision.transforms.functional as F

import pdb

def collate_fn(batch):
    return tuple(zip(*batch))


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, image, target):
        # image, annots = sample['img'], sample['annot']
        image = ((image.astype(np.float32) - self.mean) / self.std)

        return image, target


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                range(0, len(order), self.batch_size)]


class BalancedSampler(Sampler):

    def __init__(self, data_source, batch_size, train_rel=False, train_attr=False):
        # assert not (train_rel and train_attr), 'Cannot balance both relationships and attributes'
        self.data_source = data_source
        self.batch_size = batch_size
        if train_rel and train_attr:
            print('Warning! Experimental balanced sampler for joint training of attributes and relationships')
            # consider is as relationship
            rel_freq_dict = self.data_source.build_relationships_frequencies()
            attr_freq_dict = self.data_source.build_attributes_frequencies()
            attr_freq_dict = {k+self.data_source.num_relationships(): v for k, v in attr_freq_dict.items()}

            # merge the two freq dictionaries
            self.class_freq_dict = rel_freq_dict
            self.class_freq_dict.update(attr_freq_dict)
            self.num_classes = self.data_source.num_relationships() + self.data_source.num_attributes()

        elif not (train_attr or train_rel):
            self.class_freq_dict = self.data_source.build_class_frequencies()
            self.num_classes = self.data_source.num_classes()
        elif train_rel:
            self.class_freq_dict = self.data_source.build_relationships_frequencies()
            self.num_classes = self.data_source.num_relationships()
        elif train_attr:
            self.class_freq_dict = self.data_source.build_attributes_frequencies()
            self.num_classes = self.data_source.num_attributes()
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        return len(self.groups)

    def group_images(self):
        order = []
        images_per_class = len(self.data_source) // self.num_classes
        for cl, img_set in self.class_freq_dict.items():
            if len(img_set) >= images_per_class:
                # pick random images from this class
                order.extend(random.sample(img_set, images_per_class))
            else:
                # repeat indexes
                num_of_repetitions = images_per_class // len(img_set) + 1
                order.extend(list(img_set) * num_of_repetitions)

        # determine the order of the images
        # order = list(range(len(self.data_source)))
        # order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))
        random.shuffle(order)

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                range(0, len(order), self.batch_size)]
