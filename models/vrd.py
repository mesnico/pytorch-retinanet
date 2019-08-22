import torch
import torch.nn as nn
import torchvision
import itertools

import numpy as np
from torchvision.models.detection.transform import GeneralizedRCNNTransform

import pdb


class VRD(nn.Module):
    def __init__(self, detector, dataset, finetune_detector=False):
        super(VRD, self).__init__()
        self.detector = detector.module if isinstance(detector, nn.DataParallel) else detector
        # we want the detector in evaluation mode
        self.detector.training = True
        self.training = True
        self.cuda_on = False
        self.num_classes = dataset.num_classes()
        self.num_attributes = dataset.num_attributes()
        self.num_relationships = dataset.num_relationships()
        self.finetune_detector = finetune_detector

        self.attributes_classifier = nn.Sequential(
            nn.Linear(4 * 4 * 256 * 2 + self.num_classes, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, self.num_attributes),
        )

        self.relationships_classifier = nn.Sequential(
            nn.Linear(4 * 4 * 256 * 3 + 2 * self.num_classes, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, self.num_relationships),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.attr_loss_fn = nn.MultiLabelSoftMarginLoss(reduction='sum')    # multi-label classification problem
        self.rel_loss_fn = nn.CrossEntropyLoss(reduction='sum')    # standard classification problem

    def set_training(self, training=True):
        self.training = training
        self.detector.training = training

    def bbox_union(self, box1, box2, padding=0):
        x1 = min(box1[0], box2[0])
        y1 = min(box1[1], box2[1])
        x2 = max(box1[2], box2[2])
        y2 = max(box1[3], box2[3])
        # w = max(box1[0] + box1[2], box2[0] + box2[2]) - x
        # h = max(box1[1] + box1[3], box2[1] + box2[3]) - y
        out_box = torch.FloatTensor([x1 - padding, y1 - padding, x2 + padding, y2 + padding])
        if self.cuda_on:
            out_box = out_box.cuda()
        return out_box

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        losses_dict = {}

        if self.training:
            # train pass in the detector, if we want
            if self.finetune_detector:
                det_loss = self.detector(images, targets)
                losses_dict.update(det_loss)

            # transform images and targets to match the ones processed by the detector
            images, targets = self.detector.transform(images, targets)

            # objects from every batch
            objects = [t['boxes'] for t in targets]
            labels = [t['labels'] for t in targets]  # labels from every batch
        else:
            # forward through the object detector in order to retrieve objects from the image
            detections = self.detector(images)
            objects = [d['boxes'] for d in detections]
            labels = [d['labels'] for d in detections]

            # transform images and targets to match the ones processed by the detector
            images, targets = self.detector.transform(images, targets)

        # TODO: warning! Do we want gradient to be propagated into the backbone of the detector?
        image_features = self.detector.backbone(images.tensors)[3]
        image_features_pooled = self.avgpool(self.detector.backbone(images.tensors)['pool'])

        if not self.finetune_detector:
            # detach the features from the graph so that we do not backprop through the detector
            image_features = image_features.detach()
            image_features_pooled = image_features_pooled.detach()

        # iterate through batch size
        attr_loss = 0
        rel_loss = 0
        for idx, (img, img_f, img_f_pool, obj, l) in enumerate(zip(images.tensors, image_features, image_features_pooled, objects, labels)):
            # compute the scale factor between the image dimensions and the feature map dimension
            scale_factor = np.array(img_f.shape[-2:]) / np.array(img.shape[-2:])
            assert scale_factor[0] == scale_factor[1]
            scale = scale_factor[0]

            pooled_regions = torchvision.ops.roi_align(img_f.unsqueeze(0), [obj], output_size=(4, 4), spatial_scale=scale)   # K x C x H x W

            # Infer the attributes for every object in the images

            # 1. Concatenate image_features, pooled_regions and labels
            attr_features = torch.cat(
                (
                    pooled_regions.view(pooled_regions.size(0), -1),    # K x (256*4*4)
                    img_f_pool.view(-1).unsqueeze(0).expand(obj.size(0), -1),   # concatenate image level features to all the regions  K x (256*4*4)
                    nn.functional.one_hot(l, self.num_classes).float()     # K x num_classes
                ),
                dim=1
            )

            # 2. Run the multi-label classifier
            attr_out = self.attributes_classifier(attr_features)    # K x num_attr
            if self.training:
                attr_loss += self.attr_loss_fn(attr_out, targets[idx]['attributes'].float())
            else:
                raise NotImplementedError()
                # TODO: infer attributes for every detection

            # Infer the relationships between objects

            # Pseudo-code:
            # for every couple:
                # compute the union bounding box
                # pool this region in order to extract features
                # concat subj+label, rel, obj+label features
                # pass through the relationships classifier

            # 1. Compute the union bounding box
            rel_boxes = torch.zeros(obj.size(0), obj.size(0), 4)
            if self.cuda_on:
                rel_boxes = rel_boxes.cuda()
            for (idx1, box1), (idx2, box2) in itertools.permutations(enumerate(obj), 2):
                rel_boxes[idx1, idx2, :] = self.bbox_union(box1, box2, padding=20)

            # 2. Pool all the regions
            rel_boxes = rel_boxes.view(-1, 4)
            pooled_rel_regions = torchvision.ops.roi_align(img_f.unsqueeze(0), [rel_boxes], output_size=(4, 4), spatial_scale=scale)   # K x K x 256 x 4 x 4

            # 3. Prepare for the features concatenation
            pooled_obj_regions = pooled_regions.view(pooled_regions.size(0), -1).\
                unsqueeze(0).repeat(obj.size(0), 1, 1)   # K x K x 256*4*4
            pooled_subj_regions = pooled_regions.view(pooled_regions.size(0), -1).\
                unsqueeze(1).repeat(1, obj.size(0), 1)   # K x K x 256*4*4
            pooled_rel_regions = pooled_rel_regions.view(obj.size(0), obj.size(0), -1)   # K x K x 256*4*4

            # Handle labels
            one_hot_obj_label = nn.functional.one_hot(l, self.num_classes).float().unsqueeze(0).repeat(obj.size(0), 1, 1) # K x K x num_classes
            one_hot_subj_label = nn.functional.one_hot(l, self.num_classes).float().unsqueeze(1).repeat(1, obj.size(0), 1) # K x K x num_classes

            # 3. Concatenate all the features
            pooled_concat = torch.cat(
                (
                    pooled_obj_regions, one_hot_obj_label,
                    pooled_rel_regions,
                    pooled_subj_regions, one_hot_subj_label
                ),
                dim=2
            )

            # Reshape for passing through the classifier
            pooled_concat = pooled_concat.view(obj.size(0) ** 2, -1)

            # 4. Run the Relationship classifier
            rel_out = self.relationships_classifier(pooled_concat)
            if torch.isnan(rel_out).any():
                print('Rel out contain NaNs')
                pdb.set_trace()
            if self.training:
                rel_loss += self.rel_loss_fn(rel_out, targets[idx]['relationships'].view(-1))
            else:
                raise NotImplementedError()
                # TODO: infer attributes for every detection

        if self.training:
            # Compute the mean losses over all detections for every batch
            num_objects_total = sum([t['boxes'].size(0) for t in targets])
            attr_loss /= num_objects_total
            rel_loss /= num_objects_total ** 2

            losses_dict.update({'relationships_loss': rel_loss, 'attributes_loss': attr_loss})
            return losses_dict

        else:
            raise NotImplementedError
            # TODO: return boxes, attributes and relationships


    def cuda(self, device=None):
        self.cuda_on = True
        return super().cuda(device)





