import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import itertools

import numpy as np
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from models.focal_loss import FocalLoss

import pdb


class RelationshipsModelBase(nn.Module):
    def __init__(self, dataset, rel_context='relation_box', use_labels=False):
        super().__init__()

        self.num_relationships = dataset.num_relationships()
        self.num_classes = dataset.num_classes()
        self.rel_context = rel_context
        self.use_labels = use_labels
        print('Use labels: {}'.format(use_labels))

        self.rel_class_loss_fn = nn.CrossEntropyLoss(ignore_index=0)    # standard classification problem
        # self.rel_relationshipness_loss_fn = FocalLoss()
        self.rel_relationshipness_loss_fn = nn.BCEWithLogitsLoss()
        # self.rel_loss_fn = FocalLoss(num_classes=self.num_relationships, reduction='sum')

    def bbox_union(self, boxes_perm, padding=0):
        x1, _ = torch.min(boxes_perm[:, :, [0, 4]], dim=2)
        y1, _ = torch.min(boxes_perm[:, :, [1, 5]], dim=2)
        x2, _ = torch.max(boxes_perm[:, :, [2, 6]], dim=2)
        y2, _ = torch.max(boxes_perm[:, :, [3, 7]], dim=2)
        # w = max(box1[0] + box1[2], box2[0] + box2[2]) - x
        # h = max(box1[1] + box1[3], box2[1] + box2[3]) - y
        out_box = torch.stack([x1 - padding, y1 - padding, x2 + padding, y2 + padding], dim=2)
        return out_box

    def spatial_features(self, boxes_perm):
        deltax = (boxes_perm[:, :, 0] - boxes_perm[:, :, 4]) / (boxes_perm[:, :, 6] - boxes_perm[:, :, 4])
        deltay = (boxes_perm[:, :, 1] - boxes_perm[:, :, 5]) / (boxes_perm[:, :, 7] - boxes_perm[:, :, 5])
        logw = torch.log((boxes_perm[:, :, 2] - boxes_perm[:, :, 0]) / (boxes_perm[:, :, 6] - boxes_perm[:, :, 4]))
        logh = torch.log((boxes_perm[:, :, 3] - boxes_perm[:, :, 1]) / (boxes_perm[:, :, 7] - boxes_perm[:, :, 5]))
        area1 = (boxes_perm[:, :, 2] - boxes_perm[:, :, 0]) * (boxes_perm[:, :, 3] - boxes_perm[:, :, 1])
        area2 = (boxes_perm[:, :, 6] - boxes_perm[:, :, 4]) * (boxes_perm[:, :, 7] - boxes_perm[:, :, 5])

        res = torch.stack([deltax, deltay, logw, logh, area1, area2], dim=2)
        return res

    def choose_rel_indexes(self, relationships):
        # get annotated relationships and their opposite (dog is under the table and table is NOT under the dog)
        a = relationships > 0
        #b = torch.transpose(a, 0, 1)
        chosen = a  # + b

        # add some null random relationship to the set (30%)
        rand_matrix = torch.rand_like(relationships, dtype=torch.float)
        #c = rand_matrix < 0.3
        #chosen += c

        # add a random amount of null relationships
        rand_values = torch.randint(0, relationships.shape[0] ** 2, size=(torch.nonzero(relationships).shape[0],))
        d = torch.zeros_like(relationships)
        d = d.view(-1)
        d[rand_values] = 1
        d = d.view(relationships.shape[0], relationships.shape[1])
        chosen += d.byte()

        # make sure the diagonal is 0 (there are no relationships between an object and itself)
        '''not_diagonal = 1 - torch.eye(relationships.size(0))
        not_diagonal = not_diagonal.byte()
        if self.cuda_on:
            not_diagonal = not_diagonal.cuda()
        chosen = chosen * not_diagonal'''

        # at least one value should be 1 in order to avoid that the whole matrix is 0 (floating point exception happens)
        if torch.nonzero(relationships).shape[0] == 0:
            rawind = torch.argmax(rand_matrix)
            chosen[rawind // relationships.size(0), rawind % relationships.size(1)] = 1
            print('WARNING! Images with zero relationships should not be here now.')

        return chosen > 0

    def forward(self, boxes, labels, targets, img_features, pooled_regions, img_shape, scale):
        # Infer the relationships between objects

        # Pseudo-code:
        # for every couple:
        # compute the union bounding box
        # pool this region in order to extract features
        # concat subj+label, rel, obj+label features
        # pass through the relationships classifier

        # 0. Compute the union bounding box and the spatial features
        obj_perm = boxes.unsqueeze(0).repeat(boxes.size(0), 1, 1)  # K x K x 4
        subj_perm = boxes.unsqueeze(1).repeat(1, boxes.size(0), 1)  # K x K x 4
        box1box2_perm = torch.cat((obj_perm, subj_perm), dim=2)  # K x K x 8
        relboxes = self.bbox_union(box1box2_perm, padding=10)  # K x K x 4
        box1relboxes_perm = torch.cat((obj_perm, relboxes), dim=2)  # K x K x 8
        relboxesbox2_perm = torch.cat((relboxes, subj_perm), dim=2)  # K x K x 8

        box1box2_perm_feats = self.spatial_features(box1box2_perm)  # K x K x 4
        so_feats = box1box2_perm_feats[:, :, :-2]  # K x K x 2, exclude areas
        area_boxes_over_img = box1box2_perm_feats[:, :, -2:] / (
                    img_shape[0] * img_shape[1])  # take only the areas and normalize with respect to frame
        sp_feats = self.spatial_features(box1relboxes_perm)[:, :, :-2]  # K x K x 4
        po_feats = self.spatial_features(relboxesbox2_perm)[:, :, :-2]  # K x K x 4

        spatial_features = torch.cat([so_feats, sp_feats, po_feats, area_boxes_over_img], dim=2)

        # 1. Compute the union bounding box, only if rel_context is not None
        if self.rel_context == 'relation_box':
            # 2. Pool all the regions
            relboxes = relboxes.view(-1, 4)
            pooled_rel_regions = torchvision.ops.roi_align(img_features.unsqueeze(0), [relboxes], output_size=(4, 4),
                                                           spatial_scale=scale)  # K*K x 256 x 4 x 4
            # Prepare the relationship features for the concatenation
            pooled_rel_regions = pooled_rel_regions.view(boxes.size(0), boxes.size(0),
                                                         pooled_rel_regions.size(1),
                                                         pooled_rel_regions.size(2),
                                                         pooled_rel_regions.size(3))  # K x K x 256 x 4 x 4
        elif self.rel_context == 'whole_image':
            raise NotImplementedError()
            # self.avgpool(img_features)
            # TODO!
        elif self.rel_context == 'image_level_labels':
            raise NotImplementedError()
            # TODO!
        else:
            pooled_rel_regions = None

        # Stack the subject and object features
        pooled_obj_regions = pooled_regions.unsqueeze(0).repeat(boxes.size(0), 1, 1, 1, 1)  # K x K x 256 x 4 x 4
        pooled_subj_regions = pooled_regions.unsqueeze(1).repeat(1, boxes.size(0), 1, 1, 1)  # K x K x 256 x 4 x 4
        pooled_subj_obj_regions = torch.cat((pooled_subj_regions, pooled_obj_regions), dim=2)  # K x K x 512 x 4 x 4

        if self.use_labels:
            # Handle labels
            one_hot_obj_label = nn.functional.one_hot(labels, self.num_classes).float().unsqueeze(0).repeat(boxes.size(0),
                                                                                                            1,
                                                                                                            1)  # K x K x num_classes
            one_hot_subj_label = nn.functional.one_hot(labels, self.num_classes).float().unsqueeze(1).repeat(1,
                                                                                                             boxes.size(0),
                                                                                                             1)  # K x K x num_classes
            one_hot_subj_obj_label = torch.cat((one_hot_subj_label, one_hot_obj_label), dim=2)
        else:
            one_hot_subj_obj_label = None

        if self.training:
            # If training, we suppress some of the relationships
            # Hence, calculate a filter in order to control the amount of relations and non-relations seen by the architecture.
            choosen_relation_indexes = self.choose_rel_indexes(targets['relationships'])
        else:
            choosen_relation_indexes = None

        # 4. Run the Relationship classifier
        rel_out = self.features_to_relationships(pooled_subj_obj_regions, spatial_features,
                                                 one_hot_subj_obj_label, pooled_rel_regions,
                                                 choosen_relation_indexes)
        if self.training:
            t = targets['relationships'][choosen_relation_indexes]
            rel_class_loss = self.rel_class_loss_fn(rel_out, t)
            rel_relationshipness_loss = self.rel_relationshipness_loss_fn(rel_out[:, 0], (t > 0).float())

            return rel_class_loss, rel_relationshipness_loss
        else:
            inferred_rels = F.softmax(rel_out[:, 1:], dim=1)
            _, rels_indexes = torch.max(inferred_rels, dim=1)
            rels_scores = torch.sigmoid(rel_out[:, 0])  # the relationshipness is considered as score

            # reshape back to a square matrix
            rels_scores = rels_scores.view(boxes.size(0), boxes.size(0))
            rels_indexes = rels_indexes.view(boxes.size(0), boxes.size(0))
            rels_indexes += 1   # since the index 0 is the null relationship

            return {'relationships': rels_indexes, 'relationships_scores': rels_scores}

    def features_to_relationships(self, pooled_subj_obj_regions, spatial_features,
                                  one_hot_subj_obj_label, pooled_rel_regions,
                                  choosen_relation_indexes):
        # Should be overridden by the extending classes
        raise NotImplementedError()


class RelationshipsModelsSingleNet(RelationshipsModelBase):
    def __init__(self, dataset, rel_context='relation_box', use_labels=False):
        super().__init__(dataset, rel_context, use_labels)
        if rel_context == 'relation_box':
            input_size = 4 * 4 * 256 * 3 + 2 * self.num_classes * use_labels
        elif rel_context == 'whole_image':
            input_size = 4 * 4 * 256 * 3 + 2 * self.num_classes * use_labels
        elif rel_context == 'image_level_labels':
            input_size = 4 * 4 * 256 * 2 + 3 * self.num_classes * use_labels
        elif rel_context is None:
            input_size = 4 * 4 * 256 * 2 + 2 * self.num_classes * use_labels
        # add spatial feature
        input_size += 14
        self.relationships_classifier = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, self.num_relationships),
        )

    def features_to_relationships(self, pooled_subj_obj_regions, spatial_features, one_hot_subj_obj_label,
                                  pooled_rel_regions, choosen_idxs):
        # Prepare object features for concatenation
        pooled_subj_obj_regions.view(pooled_subj_obj_regions.size(0), pooled_subj_obj_regions.size(1), -1)  # K x K x 512*4*4

        # Concatenate object regions and spatial features
        pooled_concat = torch.cat((pooled_subj_obj_regions, spatial_features), dim=2)

        # If needed, concatenate object labels
        if self.use_labels:
            pooled_concat = torch.cat((pooled_concat, one_hot_subj_obj_label), dim=2)

        # If needed, concatenate the feature regarding the relationship context
        if self.rel_context is not None:
            # First, prepare the relationship features for the concatenation
            pooled_rel_regions = pooled_rel_regions.view(pooled_rel_regions.size(0), pooled_rel_regions.size(0), -1)  # K x K x 256*4*4
            pooled_concat = torch.cat((pooled_concat, pooled_rel_regions), dim=2)

        if self.training:
            # If training, we suppress some of the relationships
            # Hence, calculate a filter in order to control the amount of relations and non-relations seen by the architecture.
            pooled_concat = pooled_concat[choosen_idxs]
        else:
            # Reshape for passing through the classifier
            pooled_concat = pooled_concat.view(pooled_concat.size(0) ** 2, -1)

        # 4. Run the Relationship classifier
        rel_out = self.relationships_classifier(pooled_concat)
        return rel_out


class RelationshipsModelsMultipleNets(RelationshipsModelBase):
    def __init__(self, dataset, rel_context='relation_box', use_labels=False):
        super().__init__(dataset, rel_context, use_labels)
        if rel_context == 'relation_box':
            self.context_net = nn.Sequential(
                nn.Conv2d(256, 512, 2, stride=2),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Dropout(),
            )
        elif rel_context == 'whole_image':
            self.context_net = nn.Sequential(
                nn.Conv2d(256, 512, 2, stride=2),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Dropout(),
            )
        elif rel_context == 'image_level_labels':
            raise NotImplementedError
        elif rel_context is None:
            self.context_net = None

        self.spatial_net = nn.Sequential(
            nn.Linear(14, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.objects_convnet = nn.Sequential(
            nn.Conv2d(512, 1024, 2, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(),
        )

        if use_labels:
            self.labels_net = nn.Sequential(
                nn.Linear(2 * self.num_classes, 256),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(),
            )

        # final classifier
        input = 256 + 1024    # spatial features + objects
        if use_labels:
            input += 256
        if rel_context == 'relation_box' or rel_context == 'whole_image':
            input += 512
        self.final_classifier = nn.Sequential(
            nn.Linear(input, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.num_relationships),
        )

    def features_to_relationships(self, pooled_subj_obj_regions, spatial_features, one_hot_subj_obj_label,
                                  pooled_rel_regions, choosen_idxs):
        if self.training:
            # Filter training examples

            pooled_subj_obj_regions = pooled_subj_obj_regions[choosen_idxs]
            spatial_features = spatial_features[choosen_idxs]
            if self.use_labels:
                one_hot_subj_obj_label = one_hot_subj_obj_label[choosen_idxs]
            if self.rel_context is not None:
                pooled_rel_regions = pooled_rel_regions[choosen_idxs]
        else:
            # Reshape these tensors in order to pass through the net
            pooled_subj_obj_regions = pooled_subj_obj_regions.view(-1, pooled_subj_obj_regions.size(2),
                                                                   pooled_subj_obj_regions.size(3),
                                                                   pooled_subj_obj_regions.size(4))
            spatial_features = spatial_features.view(-1, spatial_features.size(2))
            if self.use_labels:
                one_hot_subj_obj_label = one_hot_subj_obj_label.view(-1, one_hot_subj_obj_label.size(2))
            if self.rel_context is not None:
                pooled_rel_regions = pooled_rel_regions.view(-1, pooled_rel_regions.size(2),
                                                             pooled_rel_regions.size(3),
                                                             pooled_rel_regions.size(4))

        # Forward through the net
        p_spatial = self.spatial_net(spatial_features)

        p_objects = self.objects_convnet(pooled_subj_obj_regions)
        p_objects = p_objects.mean(dim=(2, 3))  # global average pooling

        concat = torch.cat((p_objects, p_spatial), dim=1)
        if self.use_labels:
            # Forward the label net
            p_labels = self.labels_net(one_hot_subj_obj_label)
            concat = torch.cat((concat, p_labels), dim=1)

        if self.rel_context is not None:
            # Forward the context network
            p_context = self.context_net(pooled_rel_regions)
            p_context = p_context.mean(dim=(2, 3))
            concat = torch.cat((concat, p_context), dim=1)

        rel_out = self.final_classifier(concat)
        return rel_out


class AttributesModelBase(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.num_attributes = dataset.num_attributes()
        self.num_classes = dataset.num_classes()
        self.attr_loss_fn = nn.MultiLabelSoftMarginLoss()  # multi-label classification problem
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, boxes, labels, targets, img_features, pooled_regions):
        # Infer the attributes for every object in the images

        one_hot_label = nn.functional.one_hot(labels, self.num_classes)

        # 2. Run the multi-label classifier
        attr_out = self.features_to_attributes(img_features, pooled_regions, one_hot_label)
        if self.training:
            attr_loss = self.attr_loss_fn(attr_out, targets['attributes'].float())
            return attr_loss
        else:
            inferred_attr = torch.sigmoid(attr_out)
            attr_scores, attr_indexes = torch.sort(inferred_attr, dim=1, descending=True)
            return {'attributes': attr_indexes, 'attributes_scores': attr_scores}

    def features_to_attributes(self, img_features, pooled_regions, one_hot_label):
        raise NotImplementedError


class AttributesModelSingleNet(AttributesModelBase):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.attributes_classifier = nn.Sequential(
            nn.Linear(4 * 4 * 256 + self.num_classes, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, self.num_attributes),
        )

    def features_to_attributes(self, img_features, pooled_regions, one_hot_label):
        # Compute global image features
        # img_features_pooled = self.avgpool(img_features)
        # 1. Concatenate image_features, pooled_regions and labels
        attr_features = torch.cat(
            (
                pooled_regions.view(pooled_regions.size(0), -1),  # K x (256*4*4)
                # img_features_pooled.view(-1).unsqueeze(0).expand(img_features_pooled.size(0), -1),
                # concatenate image level features to all the regions  K x (256*4*4)
                one_hot_label.float()  # K x num_classes
            ),
            dim=1
        )
        out = self.attributes_classifier(attr_features)  # K x num_attr
        return out


class AttributesModelMultipleNets(AttributesModelBase):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.final_classifier = nn.Sequential(
            nn.Linear(512 + 256, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.num_attributes),
        )

        self.labels_net = nn.Sequential(
            nn.Linear(self.num_classes, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.objects_convnet = nn.Sequential(
            nn.Conv2d(256, 512, 2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(),
        )

    def features_to_attributes(self, img_features, pooled_regions, one_hot_label):
        # Process labels
        p_label = self.labels_net(one_hot_label)

        p_objects = self.objects_convnet(pooled_regions)
        p_objects = p_objects.mean(dim=(2, 3))

        concat = torch.cat((p_label, p_objects), dim=1)

        out = self.final_classifier(concat)  # K x num_attr
        return out


class VRD(nn.Module):
    def __init__(self, detector, dataset, finetune_detector=False, train_relationships=True,
                 train_attributes=True, rel_context='relation_box', use_labels=True, max_objects=80, lam=1):
        super(VRD, self).__init__()

        # asserts
        assert train_relationships or train_attributes, "You have to train one of relationships or attributes!"
        assert not (rel_context is None and train_relationships), "You have to specify a valid rel_context!"

        self.detector = detector.module if isinstance(detector, nn.DataParallel) else detector
        self.cuda_on = False
        self.num_classes = dataset.num_classes()
        self.finetune_detector = finetune_detector
        self.rel_context = rel_context
        self.train_relationships = train_relationships
        self.train_attributes = train_attributes
        self.max_objects = max_objects
        self.lam = lam

        if train_relationships:
            self.relationships_net = RelationshipsModelsMultipleNets(dataset, rel_context, use_labels)
        if train_attributes:
            self.attributes_net = AttributesModelMultipleNets(dataset)

    def train(self, mode=True):
        self.detector.train(mode)
        return super().train(mode)

    def eval(self):
        self.detector.eval()
        return super().eval()

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        losses_dict = {}
        vrd_detections = []

        if self.training:
            # train pass in the detector, if we want
            if self.finetune_detector:
                det_loss = self.detector(images, targets)
                losses_dict.update(det_loss)

            # transform images and targets to match the ones processed by the detector
            images, targets = self.detector.transform(images, targets)

            # objects from every batch
            boxes = [t['boxes'] for t in targets]
            labels = [t['labels'] for t in targets]  # labels from every batch
        else:
            # forward through the object detector in order to retrieve objects from the image
            detections = self.detector(images)
            boxes = [d['boxes'] for d in detections]
            labels = [d['labels'] for d in detections]

            # transform images and targets to match the ones processed by the detector
            images, targets = self.detector.transform(images, targets)

        image_features = self.detector.backbone(images.tensors)[3]

        if not self.finetune_detector:
            # detach the features from the graph so that we do not backprop through the detector
            image_features = image_features.detach().clone()

        # iterate through batch size
        attr_loss = 0
        rel_class_loss = 0
        rel_relationshipness_loss = 0
        for idx, (img, img_f, b, l) in enumerate(zip(images.tensors, image_features, boxes, labels)):
            # if evaluating and no objects are detected, return empty tensors
            if not self.training and b.shape[0] == 0:
                dummy_tensor = torch.FloatTensor([[0]])
                if self.cuda_on:
                    dummy_tensor = dummy_tensor.cuda()
                vrd_detections.append({'relationships': dummy_tensor, 'relationships_scores': dummy_tensor,
                                       'attributes': dummy_tensor, 'attributes_scores': dummy_tensor})
                break

            # Hard limit detected objects
            limit = self.max_objects
            how_many = b.size(0)
            if how_many > limit:
                b = b[:limit]
                l = l[:limit]
                if targets is not None:
                    targets[idx]['labels'] =  targets[idx]['labels'][:limit]
                    targets[idx]['relationships'] = targets[idx]['relationships'][:limit, :limit]
                    targets[idx]['attributes'] = targets[idx]['attributes'][:limit, :]
                # print('Skipping... too many objects ({})'.format(how_many))

            # compute the scale factor between the image dimensions and the feature map dimension
            scale_factor = np.array(img_f.shape[-2:]) / np.array(img.shape[-2:])
            assert scale_factor[0] == scale_factor[1]
            scale = scale_factor[0]

            pooled_regions = torchvision.ops.roi_align(img_f.unsqueeze(0), [b],
                                                       output_size=(4, 4), spatial_scale=scale)   # K x C x H x W

            # Prepare targets if needed (during training)
            t = targets[idx] if self.training else None

            vrd_detection_dict = {}

            # Train or Infer relationships
            if self.train_relationships:
                out_rel = self.relationships_net(b, l, t, img_f, pooled_regions, img.shape[-2:], scale)
                if self.training:
                    # out_rel contains a loss value
                    rel_class_loss += out_rel[0]
                    rel_relationshipness_loss += out_rel[1]
                else:
                    # out_rel contains detections
                    vrd_detection_dict = out_rel

            # Train or Infer attributes
            if self.train_attributes:
                out_attr = self.attributes_net(b, l, t, img_f, pooled_regions)
                if self.training:
                    # out_attr contains a loss value
                    attr_loss += out_attr
                else:
                    # out_attr contains detections
                    vrd_detection_dict.update(out_attr)

            if not self.training:
                vrd_detections.append(vrd_detection_dict)

        if self.training:
            # Compute the mean losses over all detections for every batch
            #num_objects_total = sum([t['boxes'].size(0) for t in targets])
            attr_loss /= len(images.tensors)
            rel_class_loss /= len(images.tensors)
            rel_relationshipness_loss /= len(images.tensors)

            if self.train_relationships:
                losses_dict.update({'relationships_class_loss': self.lam * rel_class_loss})
                losses_dict.update({'relationshipness_loss': rel_relationshipness_loss})
            if self.train_attributes:
                losses_dict.update({'attributes_loss': attr_loss})

            return losses_dict

        else:
            # Merge boxes, inferred_attr and inferred_rels using the same interface of the detection in torchvision
            detections = [{**obj_det, **vrd_det} for obj_det, vrd_det in zip(detections, vrd_detections)]
            return detections

    def cuda(self, device=None):
        self.cuda_on = True
        return super().cuda(device)





