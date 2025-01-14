from __future__ import print_function, division

import csv
import pickle
import os
import warnings
import tqdm
from progressbar import *

import numpy as np
import skimage
import skimage.color
import skimage.io
import skimage.transform
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import shelve
import itertools
import scipy

import pdb

def get_labels(metadata_dir, version='v5'):
    if version == 'v4' or version == 'v5' or version == 'challenge2018':
        csv_file = 'class-descriptions-boxable.csv' if version == 'v4' or version == 'v5' else 'challenge-2018-class-descriptions-500.csv'

        boxable_classes_descriptions = os.path.join(metadata_dir, csv_file)
        id_to_labels = {}
        id_to_labels_idx = {}
        cls_index = {}

        i = 1
        with open(boxable_classes_descriptions) as f:
            for row in csv.reader(f):
                # make sure the csv row is not empty (usually the last one)
                if len(row):
                    label = row[0]
                    description = row[1].replace("\"", "").replace("'", "").replace('`', '')

                    id_to_labels_idx[i] = label
                    id_to_labels[i] = description
                    cls_index[label] = i

                    i += 1

        # Add background class
        id_to_labels[0] = '__background__'
        id_to_labels_idx[0] = '/m/back'
        cls_index['/m/back'] = 0

    else:
        raise NotImplementedError()

    return id_to_labels, id_to_labels_idx, cls_index


def get_attribute_relationships_labels(metadata_dir, version='v5'):
    if version == 'v4' or version == 'v5' or version == 'challenge2018':
        attributes_csv_file = 'challenge-2018-attributes-description.csv'
        relationships_csv_file = 'challenge-2018-relationships-description.csv'

        attribute_descriptions = os.path.join(metadata_dir, attributes_csv_file)
        relationship_descriptions = os.path.join(metadata_dir, relationships_csv_file)
        attr_id_to_labels = {}
        attr_id_to_labels_idx = {}
        attr_index = {}
        rel_id_to_labels = {}
        rel_id_to_labels_idx = {}
        rel_index = {}

        # Handle attributes
        i = 1
        with open(attribute_descriptions) as f:
            for row in csv.reader(f):
                # make sure the csv row is not empty (usually the last one)
                if len(row):
                    label = row[0]
                    description = row[1].replace("\"", "").replace("'", "").replace('`', '')

                    attr_id_to_labels_idx[i] = label
                    attr_id_to_labels[i] = description
                    attr_index[label] = i

                    i += 1

        # Add non class to attributes
        attr_id_to_labels[0] = '__none__'
        attr_id_to_labels_idx[0] = '/m/none'
        attr_index['/m/none'] = 0

        # Handle relationships
        i = 1
        with open(relationship_descriptions) as f:
            for row in csv.reader(f):
                # make sure the csv row is not empty (usually the last one)
                if len(row):
                    label = row[0]
                    description = row[1].replace("\"", "").replace("'", "").replace('`', '')
                    if description == 'is':
                        # 'is' is not a real relationship
                        continue

                    rel_id_to_labels_idx[i] = label
                    rel_id_to_labels[i] = description
                    rel_index[label] = i

                    i += 1

        # Add non class to relationships
        rel_id_to_labels[0] = '__none__'
        rel_id_to_labels_idx[0] = '/m/none'
        rel_index['/m/none'] = 0

    else:
        raise NotImplementedError()

    return attr_id_to_labels, attr_id_to_labels_idx, attr_index, rel_id_to_labels, rel_id_to_labels_idx, rel_index


def generate_images_annotations_json(main_dir, metadata_dir, subset, cls_index, version='v5'):
    validation_image_ids = {}

    if version == 'v4' or version == 'v5':
        annotations_path = os.path.join(metadata_dir, '{}-annotations-bbox.csv'.format(subset))
    elif version == 'challenge2018':
        validation_image_ids_path = os.path.join(metadata_dir, 'challenge-2018-image-ids-valset-od.csv')

        with open(validation_image_ids_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file, fieldnames=['ImageID'])
            reader.next()
            for line, row in enumerate(reader):
                image_id = row['ImageID']
                validation_image_ids[image_id] = True

        annotations_path = os.path.join(metadata_dir, 'challenge-2018-train-annotations-bbox.csv')
    else:
        annotations_path = os.path.join(metadata_dir, subset, 'annotations-human-bbox.csv')

    fieldnames = ['ImageID', 'Source', 'LabelName', 'Confidence',
                  'XMin', 'XMax', 'YMin', 'YMax',
                  'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside']

    id_annotations = dict()
    if os.path.exists(annotations_path):
        with open(annotations_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file, fieldnames=fieldnames)
            next(reader)

            images_sizes = {}
            for line, row in enumerate(tqdm.tqdm(reader)):
                frame = row['ImageID']

                if version == 'challenge2018':
                    if subset == 'train':
                        if frame in validation_image_ids:
                            continue
                    elif subset == 'validation':
                        if frame not in validation_image_ids:
                            continue
                    else:
                        raise NotImplementedError('This generator handles only the train and validation subsets')

                class_name = row['LabelName']

                if class_name not in cls_index:
                    continue

                cls_id = cls_index[class_name]

                if version == 'challenge2018':
                    # We recommend participants to use the provided subset of the training set as a validation set.
                    # This is preferable over using the V4 val/test sets, as the training set is more densely annotated.
                    img_path = os.path.join(main_dir, 'train', frame + '.jpg')
                else:
                    img_path = os.path.join(main_dir, subset, frame + '.jpg')

                if frame in images_sizes:
                    width, height = images_sizes[frame]
                else:
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.width, img.height
                            images_sizes[frame] = (width, height)
                    except Exception as ex:
                        if version == 'challenge2018':
                            raise ex
                        continue

                x1 = float(row['XMin'])
                x2 = float(row['XMax'])
                y1 = float(row['YMin'])
                y2 = float(row['YMax'])

                x1_int = int(round(x1 * width))
                x2_int = int(round(x2 * width))
                y1_int = int(round(y1 * height))
                y2_int = int(round(y2 * height))

                # Check that the bounding box is valid.
                if x2 <= x1:
                    raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
                if y2 <= y1:
                    raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

                if y2_int == y1_int:
                    warnings.warn('filtering line {}: rounding y2 ({}) and y1 ({}) makes them equal'.format(line, y2, y1))
                    continue

                if x2_int == x1_int:
                    warnings.warn('filtering line {}: rounding x2 ({}) and x1 ({}) makes them equal'.format(line, x2, x1))
                    continue

                img_id = row['ImageID']
                annotation = {'cls_id': cls_id, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}

                if img_id in id_annotations:
                    annotations = id_annotations[img_id]
                    annotations['boxes'].append(annotation)
                else:
                    id_annotations[img_id] = {'w': width, 'h': height, 'boxes': [annotation]}
    else:
        # simply cache image informations from the image folder.
        # This is needed for test detections for challenge submission
        print('WARNING: annotation file not present! Supposing test dataset without annotations')
        images_fld = os.path.join(main_dir, subset)
        for image in tqdm.tqdm(os.listdir(images_fld)):
            img_id = os.path.splitext(image)[0]
            img_path = os.path.join(images_fld, image)
            with Image.open(img_path) as img:
                width, height = img.width, img.height

            # dummy annotation
            annotation = {'cls_id': 0, 'x1': 0, 'x2': 0, 'y1': 0, 'y2': 0}
            id_annotations[img_id] = {'w': width, 'h': height, 'boxes': [annotation]}

    return id_annotations


def generate_images_annotations_vrd_json(main_dir, metadata_dir, subset, cls_index, attr_index, rel_index, version='v5'):

    if version == 'v4' or version == 'v5' or version == 'challenge2018':
        filename = 'challenge-2018-train-vrd.csv' if subset == 'train' else '{}-annotations-vrd.csv'.format(subset)
        annotations_path = os.path.join(metadata_dir, filename)

    fieldnames = ['ImageID', 'LabelName1', 'LabelName2', 'XMin1', 'XMax1', 'YMin1', 'YMax1', 'XMin2', 'XMax2', 'YMin2',
                  'YMax2', 'RelationshipLabel']

    # load the box annotations
    box_annotations = cache_annotations(subset=subset, annotation_fn=generate_images_annotations_json,
                                        annotation_cache_dir='annotations_cache',
                                        kwargs=dict(main_dir=main_dir, metadata_dir=metadata_dir,
                                                    cls_index=cls_index, version=version)
                                        )

    id_annotations = dict()
    if os.path.exists(annotations_path):
        with open(annotations_path, 'r') as csv_file:
            vrd_annotations = pd.read_csv(csv_file)
            maxval = len((vrd_annotations['ImageID'].unique()))
            pbar = ProgressBar(widgets=[Percentage(), Bar(), AdaptiveETA()], maxval=maxval).start()
            now = 0

            for frame, group in vrd_annotations.groupby('ImageID'):
                if frame not in box_annotations:
                    print('Warning! Frame {} not in the object detection dictionary'.format(frame))
                    continue
                # search for all the bounding boxes in the image
                num_boxes = len(box_annotations[frame]['boxes'])
                all_frame_boxes = np.zeros((num_boxes, 5))
                all_frame_labels = np.zeros(num_boxes)
                for idx, f in enumerate(box_annotations[frame]['boxes']):
                    all_frame_boxes[idx, 0] = f['x1']
                    all_frame_boxes[idx, 1] = f['y1']
                    all_frame_boxes[idx, 2] = f['x2']
                    all_frame_boxes[idx, 3] = f['y2']
                    all_frame_boxes[idx, 4] = f['cls_id']

                rel_boxes = group.loc[:, ["XMin1", "YMin1", "XMax1", "YMax1", "XMin2", "YMin2", "XMax2", "YMax2"]].values.astype(np.float32)
                rel_obj_labels = group.loc[:, ["LabelName1", "LabelName2"]]
                rel_labels = group.loc[:, ["RelationshipLabel"]]
                # insert also class information, otherwise there are problems in case of overlapping boxes
                cls_and_attr_index = {**cls_index, **attr_index}
                rel_boxes = np.insert(rel_boxes, 4, [cls_and_attr_index[i] for i in rel_obj_labels.values[:, 0]], axis=1)
                rel_boxes = np.insert(rel_boxes, 9, [cls_and_attr_index[i] for i in rel_obj_labels.values[:, 1]], axis=1)
                rel_boxes_int = (rel_boxes * 100).astype(np.int)

                # Handle relationships
                relationships = np.zeros((num_boxes, num_boxes))

                for (idx1, box1), (idx2, box2) in itertools.permutations(enumerate(all_frame_boxes), 2):
                    actual_couple_int = (np.concatenate((box1, box2), axis=None) * 100).astype(np.int)

                    found_idxs = (rel_boxes_int == actual_couple_int).all(axis=1).nonzero()[0]
                    # found_idx = rel_boxes_int_list.index(list(actual_couple_int))
                    if len(found_idxs) > 0:
                        # this is a relationship match
                        actual_rel_labels = rel_labels.values[found_idxs, 0]
                        '''# it is possible that two different boxes have the same coordinates.
                        # Ensure that the 'is' relationship, in this case, is being excluded
                        actual_rel_labels = actual_rel_labels[actual_rel_labels != 'is']
                        if len(actual_rel_labels) > 0:
                            pdb.set_trace()
                            # get the relation that is not the 'is' one
                            actual_rel_label = actual_rel_labels[0]
                            actual_rel_id = rel_index[actual_rel_label]
                        else:
                            actual_rel_id = 0
                        '''
                        # There should be no 'is' relationships at this point, so the first relation that matches
                        # should be the right one
                        actual_rel_label = actual_rel_labels[0]
                        assert actual_rel_label != 'is', "Should NOT be an is relationship!"
                        actual_rel_id = rel_index[actual_rel_label]
                    else:
                        # this couple is not linked
                        actual_rel_id = 0

                    relationships[idx1, idx2] = actual_rel_id

                # Handle attributes
                # Create a one-hot coding for each object in the frame.
                attributes = np.zeros((num_boxes, len(attr_index)))
                for idx, box in enumerate(all_frame_boxes):
                    actual_couple_int = (np.concatenate((box, box), axis=None) * 100).astype(np.int)

                    # find all the indexes of the objects having at least one attribute
                    found_idxs = (rel_boxes_int[:, :9] == actual_couple_int[:9]).all(axis=1).nonzero()[0]
                    if len(found_idxs) > 0:
                        # this object has possibly many attributes
                        actual_rel_labels = rel_labels.values[found_idxs, 0]
                        # filter spurious entries that are not 'is'
                        filtered = actual_rel_labels == 'is'
                        attributes_labels = rel_obj_labels.values[found_idxs[filtered], 1]
                        attributes_id = [attr_index[i] for i in attributes_labels]

                        attributes[idx, attributes_id] = 1

                    # else:
                        # do nothing, this attributes row is already full of zeros

                # sparsify relationships and attributes for space efficiency
                relationships = scipy.sparse.csr_matrix(relationships)
                attributes = scipy.sparse.csr_matrix(attributes)

                id_annotations[frame] = {'w': box_annotations[frame]['w'], 'h': box_annotations[frame]['h'],
                                         'boxes': box_annotations[frame]['boxes'],
                                         'relationships': relationships, 'attributes': attributes}
                now += 1
                pbar.update(now)
    else:
        # simply cache image informations from the image folder.
        # This is needed for test detections for challenge submission
        print('WARNING: annotation file not present! Supposing test dataset without annotations')
        images_fld = os.path.join(main_dir, subset)
        for image in tqdm.tqdm(os.listdir(images_fld)):
            img_id = os.path.splitext(image)[0]
            img_path = os.path.join(images_fld, image)
            with Image.open(img_path) as img:
                width, height = img.width, img.height

            # dummy annotations
            boxes = {'cls_id': 0, 'x1': 0, 'x2': 0, 'y1': 0, 'y2': 0}
            attributes = scipy.sparse.csr_matrix(np.zeros((1,1)))
            relationships = scipy.sparse.csr_matrix(np.zeros((1,1)))
            id_annotations[img_id] = {'w': width, 'h': height, 'boxes': [boxes], 'attributes': attributes,
                                      'relationships': relationships}

    return id_annotations


def cache_annotations(subset, annotation_fn, all_in_memory=True, annotation_cache_dir='annotations_cache', kwargs={}):
    ext = '.pkl' if all_in_memory else '.db'
    annotation_cache_filename = os.path.join(annotation_cache_dir, subset + ext)
    if os.path.exists(annotation_cache_filename + '.dat') or os.path.exists(annotation_cache_filename):
        print('Loading cached annotations: {}'.format(annotation_cache_filename))
        if all_in_memory:
            with open(annotation_cache_filename, 'rb') as f:
                annotations = pickle.load(f)
        else:
            annotations = shelve.open(annotation_cache_filename)

    else:
        print('Caching annotations to file: {}'.format(annotation_cache_filename))
        annotations = annotation_fn(subset=subset, **kwargs)
        if all_in_memory:
            with open(annotation_cache_filename, "wb") as f:
                pickle.dump(annotations, f)
        else:
            with shelve.open(annotation_cache_filename) as f:
                for id, ann in annotations.items():
                    f[id] = ann
            annotations = shelve.open(annotation_cache_filename)

    return annotations


class OidDataset(Dataset):
    """Oid dataset."""

    def __init__(self, main_dir, subset, version='v5', annotation_cache_dir='annotations_cache', transform=None, all_in_memory=True):
        if version == 'v4':
            metadata = '2018_04'
        elif version == 'challenge2018':
            metadata = 'challenge2018'
        elif version == 'v3':
            metadata = '2017_11'
        elif version == 'v5':
            metadata = 'metadata'
        else:
            raise NotImplementedError('There is currently no implementation for versions older than v3')

        self.transform = transform

        if version == 'challenge2018':
            self.base_dir = os.path.join(main_dir, 'images', 'train')
        else:
            self.base_dir = os.path.join(main_dir, subset)

        metadata_dir = os.path.join(main_dir, metadata)
        ext = '.pkl' if all_in_memory else '.db'
        annotation_cache_filename = os.path.join(annotation_cache_dir, subset + ext)

        self.id_to_labels, self.id_to_labels_idx, cls_index = get_labels(metadata_dir, version=version)

        self.annotations = cache_annotations(subset=subset, annotation_fn=generate_images_annotations_json,
                                             annotation_cache_dir='annotations_cache',
                                             kwargs=dict(main_dir=main_dir, metadata_dir=metadata_dir,
                                                         cls_index=cls_index, version=version)
                                             )

        self.id_to_image_id = dict([(i, k) for i, k in enumerate(self.annotations)])

        # (label -> name)
        self.labels = self.id_to_labels

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        # sample = {'img': img, 'annot': annot}
        target = {}
        target['boxes'] = annot[:, :4]
        target['labels'] = annot[:, 4]
        if self.transform:
            img, target = self.transform(img, target)

        return img, target

    def image_path(self, image_index):
        path = os.path.join(self.base_dir, self.id_to_image_id[image_index] + '.jpg')
        return path

    def load_image(self, image_index):
        path = self.image_path(image_index)
        img = skimage.io.imread(path)

        if len(img.shape) == 1:
            img = img[0]

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        try:
            return img.astype(np.float32) / 255.0
        except Exception:
            print(path)
            exit(0)

    def load_annotations(self, image_index):
        # get ground truth annotations
        image_annotations = self.annotations[self.id_to_image_id[image_index]]

        labels = image_annotations['boxes']

        height, width = image_annotations['h'], image_annotations['w']

        boxes = np.zeros((len(labels), 5))
        for idx, ann in enumerate(labels):
            cls_id = ann['cls_id']
            x1 = ann['x1'] * width
            x2 = ann['x2'] * width
            y1 = ann['y1'] * height
            y2 = ann['y2'] * height

            boxes[idx, 0] = x1
            boxes[idx, 1] = y1
            boxes[idx, 2] = x2
            boxes[idx, 3] = y2
            boxes[idx, 4] = cls_id

        return boxes

    # used for aspect ratio sampler
    def image_aspect_ratio(self, image_index):
        img_annotations = self.annotations[self.id_to_image_id[image_index]]
        height, width = img_annotations['h'], img_annotations['w']
        return float(width) / float(height)

    # used for balanced sampler
    def build_class_frequencies(self):
        freq = {}
        idxs = list(range(len(self)))
        for idx in tqdm.tqdm(idxs):
            ann = self.annotations[self.id_to_image_id[idx]]
            classes = [v['cls_id'] for v in ann['boxes']]
            for c in classes:
                if c not in freq:
                    freq[c] = set()
                freq[c].add(idx)
        return freq

    def num_classes(self):
        return len(self.id_to_labels)

    def evaluate(self, all_detections, output_dir, file_identifier=""):
        """
        Evaluates detections and put the results in a file into outdir
        :param all_detections: list[image_index, list[boxes], list[labels]]
        :param output_dir: file where detection results will be stored
        :param file_identifier: optionally, a identifier for the file
        :return: optionally, a dictionary of metrics
        """

        # MODE 1 (python evaluation)
        det_dict = {
            'ImageID': [],
            'XMin': [],
            'XMax': [],
            'YMin': [],
            'YMax': [],
            'Score': [],
            'LabelName': []
        }

        for image_index, boxes, labels, scores in all_detections:
            img_annotations = self.annotations[self.id_to_image_id[image_index]]
            for box, label, score in zip(boxes, labels, scores):
                # add this detection to the dict
                det_dict['ImageID'].append(self.id_to_image_id[image_index])
                det_dict['XMin'].append(np.clip(box[0] / img_annotations['w'], 0, 1))
                det_dict['YMin'].append(np.clip(box[1] / img_annotations['h'], 0, 1))
                det_dict['XMax'].append(np.clip(box[2] / img_annotations['w'], 0, 1))
                det_dict['YMax'].append(np.clip(box[3] / img_annotations['h'], 0, 1))
                det_dict['Score'].append(score)
                det_dict['LabelName'].append(self.id_to_labels_idx[label])

        # dump dict on a csv file
        df = pd.DataFrame(det_dict)
        out_filename = os.path.join(output_dir, 'detections_{}.csv'.format(file_identifier))
        df.to_csv(out_filename, index=False, float_format='%.6f')

        # MODE 2 (challenge)

        predictions = []

        for image_index, boxes, labels, scores in all_detections:
            detections = []
            img_annotations = self.annotations[self.id_to_image_id[image_index]]
            for box, label, score in zip(boxes, labels, scores):
                # add this detection to the dict
                det_str = "{} {:f} {:f} {:f} {:f} {:f}".format(
                    self.id_to_labels_idx[label],
                    score,
                    np.clip(box[0] / img_annotations['w'], 0, 1),
                    np.clip(box[1] / img_annotations['h'], 0, 1),
                    np.clip(box[2] / img_annotations['w'], 0, 1),
                    np.clip(box[3] / img_annotations['h'], 0, 1)
                )
                detections.append(det_str)

            predictions.append(
                {'ImageID': self.id_to_image_id[image_index],
                 'PredictionString': " ".join(detections)}
            )

            # dump dict on a csv file
        df = pd.DataFrame(predictions)
        out_filename = os.path.join(output_dir, 'detections_{}_competitionformat.csv'.format(file_identifier))
        df.to_csv(out_filename, index=False, float_format='%.6f')


class OidDatasetVRD(Dataset):
    """Oid dataset."""

    def __init__(self, main_dir, subset, version='v5', transform=None):
        if version == 'v4':
            metadata = '2018_04'
        elif version == 'challenge2018':
            metadata = 'challenge2018'
        elif version == 'v3':
            metadata = '2017_11'
        elif version == 'v5':
            metadata = 'metadata'
        else:
            raise NotImplementedError('There is currently no implementation for versions older than v3')

        self.transform = transform

        if version == 'challenge2018':
            self.base_dir = os.path.join(main_dir, 'images', 'train')
        else:
            self.base_dir = os.path.join(main_dir, subset)

        metadata_dir = os.path.join(main_dir, metadata)

        self.id_to_labels, self.id_to_labels_idx, cls_index = get_labels(metadata_dir, version=version)
        self.attr_id_to_labels, \
        self.attr_id_to_labels_idx, \
        attr_index, \
        self.rel_id_to_labels, \
        self.rel_id_to_labels_idx, \
        rel_index = get_attribute_relationships_labels(metadata_dir, version=version)

        self.annotations = cache_annotations(subset=subset, annotation_fn=generate_images_annotations_vrd_json,
                                             annotation_cache_dir='annotations_cache_vrd',
                                             kwargs=dict(main_dir=main_dir, metadata_dir=metadata_dir,
                                                cls_index=cls_index, attr_index=attr_index, rel_index=rel_index,
                                                version=version)
                                             )

        self.id_to_image_id = dict([(i, k) for i, k in enumerate(self.annotations)])

        # (label -> name)
        self.labels = self.id_to_labels

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        boxes, attributes, relationships = self.load_annotations(idx)
        # sample = {'img': img, 'annot': annot}
        target = {}
        target['boxes'] = boxes[:, :4]
        target['labels'] = boxes[:, 4]

        target['attributes'] = attributes
        target['relationships'] = relationships

        if self.transform:
            img, target = self.transform(img, target)
        return img, target

    def image_path(self, image_index):
        path = os.path.join(self.base_dir, self.id_to_image_id[image_index] + '.jpg')
        return path

    def load_image(self, image_index):
        path = self.image_path(image_index)
        img = skimage.io.imread(path)

        if len(img.shape) == 1:
            img = img[0]

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        try:
            return img.astype(np.float32) / 255.0
        except Exception:
            print(path)
            exit(0)

    def load_annotations(self, image_index):
        # get ground truth annotations
        image_annotations = self.annotations[self.id_to_image_id[image_index]]

        labels = image_annotations['boxes']
        attributes = image_annotations['attributes']
        relationships = image_annotations['relationships']

        height, width = image_annotations['h'], image_annotations['w']

        boxes = np.zeros((len(labels), 5))
        for idx, ann in enumerate(labels):
            cls_id = ann['cls_id']
            x1 = ann['x1'] * width
            x2 = ann['x2'] * width
            y1 = ann['y1'] * height
            y2 = ann['y2'] * height

            boxes[idx, 0] = x1
            boxes[idx, 1] = y1
            boxes[idx, 2] = x2
            boxes[idx, 3] = y2
            boxes[idx, 4] = cls_id

        return boxes, attributes, relationships

    # used for aspect ratio sampler
    def image_aspect_ratio(self, image_index):
        img_annotations = self.annotations[self.id_to_image_id[image_index]]
        height, width = img_annotations['h'], img_annotations['w']
        return float(width) / float(height)

    def build_relationships_frequencies(self):
        freq = {}
        idxs = list(range(len(self)))
        for idx in tqdm.tqdm(idxs):
            ann = self.annotations[self.id_to_image_id[idx]]
            relationships = ann['relationships'].toarray().reshape(-1)
            for r in relationships:
                if r not in freq:
                    freq[r] = set()
                freq[r].add(idx)
        # the 0 index does not care at this point, delete
        del freq[0]
        return freq

    def build_attributes_frequencies(self):
        freq = {}
        idxs = list(range(len(self)))
        for idx in tqdm.tqdm(idxs):
            ann = self.annotations[self.id_to_image_id[idx]]
            _, attributes = np.nonzero(ann['attributes'].toarray())
            for a in attributes:
                if a not in freq:
                    freq[a] = set()
                freq[a].add(idx)
        return freq

    def num_classes(self):
        return len(self.id_to_labels)

    def num_attributes(self):
        return len(self.attr_id_to_labels)

    def num_relationships(self):
        return len(self.rel_id_to_labels)

    def evaluate(self, all_detections, output_dir, file_identifier=""):
        """
        Evaluates detections and put the results in a file into outdir

        :param all_detections: list[image_index, list[subj_boxes], list[subj_labels], list[obj_boxes], list[obj_labels], list[rel_labels], list[rel_scores]]
        :param output_dir: file where detection results will be stored
        :param file_identifier: optionally, a identifier for the file
        :return: optionally, a dictionary of metrics

        """

        # MODE 1 (python evaluation)
        '''
        det_dict = {
            'ImageID': [],
            'XMin': [],
            'XMax': [],
            'YMin': [],
            'YMax': [],
            'Score': [],
            'LabelName': []
        }

        for image_index, boxes, labels, scores in all_detections:
            img_annotations = self.annotations[self.id_to_image_id[image_index]]
            for box, label, score in zip(boxes, labels, scores):
                # add this detection to the dict
                det_dict['ImageID'].append(self.id_to_image_id[image_index])
                det_dict['XMin'].append(np.clip(box[0] / img_annotations['w'], 0, 1))
                det_dict['YMin'].append(np.clip(box[1] / img_annotations['h'], 0, 1))
                det_dict['XMax'].append(np.clip(box[2] / img_annotations['w'], 0, 1))
                det_dict['YMax'].append(np.clip(box[3] / img_annotations['h'], 0, 1))
                det_dict['Score'].append(score)
                det_dict['LabelName'].append(self.id_to_labels_idx[label])

        # dump dict on a csv file
        df = pd.DataFrame(det_dict)
        out_filename = os.path.join(output_dir, 'detections_{}.csv'.format(file_identifier))
        df.to_csv(out_filename, index=False, float_format='%.6f')

        '''
        # MODE 2 (challenge)

        predictions = []
        # handle attributes
        for image_index, s_boxes, s_labels, o_boxes, o_labels, r_labels, r_scores in all_detections:
            detections = []
            img_annotations = self.annotations[self.id_to_image_id[image_index]]

            for s_box, s_label, o_box, o_label, r_label, r_score in zip(s_boxes, s_labels, o_boxes, o_labels, r_labels, r_scores):
                # add this detection to the dict
                det_str = "{:f} {} {:f} {:f} {:f} {:f} {} {:f} {:f} {:f} {:f} {}".format(
                    r_score,
                    self.id_to_labels_idx[s_label],
                    np.clip(s_box[0] / img_annotations['w'], 0, 1),
                    np.clip(s_box[1] / img_annotations['h'], 0, 1),
                    np.clip(s_box[2] / img_annotations['w'], 0, 1),
                    np.clip(s_box[3] / img_annotations['h'], 0, 1),
                    self.id_to_labels_idx[o_label] if r_label is not -1 else self.attr_id_to_labels_idx[o_label],
                    np.clip(o_box[0] / img_annotations['w'], 0, 1),
                    np.clip(o_box[1] / img_annotations['h'], 0, 1),
                    np.clip(o_box[2] / img_annotations['w'], 0, 1),
                    np.clip(o_box[3] / img_annotations['h'], 0, 1),
                    self.rel_id_to_labels_idx[r_label] if r_label is not -1 else 'is'
                )
                detections.append(det_str)

            predictions.append(
                {'ImageID': self.id_to_image_id[image_index],
                 'PredictionString': " ".join(detections)}
            )

            # dump dict on a csv file
        df = pd.DataFrame(predictions)
        out_filename = os.path.join(output_dir, 'detections_{}_competitionformat.csv'.format(file_identifier))
        df.to_csv(out_filename, index=False, float_format='%.6f')


if __name__ == '__main__':
    #from dataloader import BalancedSampler
    #import sys
    #sys.path.append('..')
    from dataloader import collate_fn
    from torch.utils.data import DataLoader
    from collections import Counter

    dataset_train = OidDatasetVRD('/media/nicola/SSD/Datasets/OpenImages', subset='train')
    # sampler = BalancedSampler(dataset_train, batch_size=4, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=8, batch_size=1, collate_fn=collate_fn)

    attr_list = []
    rel_list = []
    imgs_with_no_rels = 0
    imgs_with_no_attrs = 0
    imgs_with_no_attrs_rels = 0

    ### Calculate some stats
    '''for idx, (_, target) in enumerate(tqdm.tqdm(dataloader_train)):
        # calculate some vrd stat
        na = False
        nr = False
        target = target[0]
        attributes = np.asarray(target['attributes'].todense(), dtype=np.int).reshape(-1)
        relationships = np.asarray(target['relationships'].todense(), dtype=np.int).reshape(-1)

        attr_list.extend(list(attributes))
        rel_list.extend(list(relationships))

        if np.all(attributes == 0):
            imgs_with_no_attrs += 1
            na = True
        if np.all(relationships == 0):
            imgs_with_no_rels += 1
            nr = True
        if na and nr:
            imgs_with_no_attrs_rels += 1

    print('Relations frequencies: {}'.format(Counter(rel_list)))
    print('Attributes frequencies: {}'.format(Counter(attr_list)))
    print('Num images with no attributes: {}'.format(imgs_with_no_attrs))
    print('Num images with no relationships: {}'.format(imgs_with_no_rels))
    print('Num images with both no attributes and relationships: {}'.format(imgs_with_no_attrs_rels))'''

    ### Test frequencies methods
    print('Attributes Frequencies')
    freq = dataset_train.build_attributes_frequencies()
    for k,v in freq.items():
        print('{}: {}'.format(k, len(v)))

    print('Relationships Frequencies')
    freq = dataset_train.build_relationships_frequencies()
    for k, v in freq.items():
        print('{}: {}'.format(k, len(v)))



