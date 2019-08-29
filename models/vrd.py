import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import itertools

import numpy as np
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from models.focal_loss import FocalLoss

import pdb


class VRD(nn.Module):
    def __init__(self, detector, dataset, finetune_detector=False, rel_context='relation_box'):
        super(VRD, self).__init__()
        self.detector = detector.module if isinstance(detector, nn.DataParallel) else detector
        self.detector.training = True
        self.training = True
        self.cuda_on = False
        self.num_classes = dataset.num_classes()
        self.num_attributes = dataset.num_attributes()
        self.num_relationships = dataset.num_relationships()
        self.finetune_detector = finetune_detector
        self.rel_context = rel_context

        # self.multiprocessing_pool = mp.Pool(8)

        self.attributes_classifier = nn.Sequential(
            nn.Linear(4 * 4 * 256 * 2 + self.num_classes, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, self.num_attributes),
        )

        if rel_context == 'relation_box':
            input_size = 4 * 4 * 256 * 3 + 2 * self.num_classes
        elif rel_context == 'whole_image':
            input_size = 4 * 4 * 256 * 3 + 2 * self.num_classes
        elif rel_context == 'image_level_labels':
            input_size = 4 * 4 * 256 * 2 + 3 * self.num_classes
        elif rel_context is None:
            input_size = 4 * 4 * 256 * 2 + 2 * self.num_classes
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

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.attr_loss_fn = nn.MultiLabelSoftMarginLoss(reduction='sum')    # multi-label classification problem
        # self.rel_loss_fn = nn.CrossEntropyLoss(reduction='sum')    # standard classification problem
        self.rel_loss_fn = FocalLoss(num_classes=self.num_relationships, reduction='sum')

    def train(self, mode=True):
        self.detector.train(mode)
        return super().train(mode)

    def eval(self):
        self.detector.eval()
        return super().eval()

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
        b = torch.transpose(a, 0, 1)
        chosen = a + b

        # add some null random relationship to the set (10%)
        rand_matrix = torch.rand_like(relationships, dtype=torch.float)
        c = rand_matrix < 0.3
        chosen += c

        # make sure the diagonal is 0 (there are no relationships between an object and itself)
        not_diagonal = 1 - torch.eye(relationships.size(0))
        not_diagonal = not_diagonal.byte()
        if self.cuda_on:
            not_diagonal = not_diagonal.cuda()
        chosen = chosen * not_diagonal

        # at least one value should be 1 in order to avoid that the whole matrix is 0 (floating point exception happens)
        rawind = torch.argmax(rand_matrix)
        indexes = torch.LongTensor([rawind // relationships.size(0), rawind % relationships.size(1)])
        chosen[indexes] = 1

        return chosen > 0

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
            objects = [t['boxes'] for t in targets]
            labels = [t['labels'] for t in targets]  # labels from every batch
        else:
            # forward through the object detector in order to retrieve objects from the image
            detections = self.detector(images)
            objects = [d['boxes'] for d in detections]
            labels = [d['labels'] for d in detections]

            # transform images and targets to match the ones processed by the detector
            images, targets = self.detector.transform(images, targets)

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
            # if evaluating and no objects are detected, return empty tensors
            if not self.training and obj.shape[0] == 0:
                dummy_tensor = torch.FloatTensor([[0]])
                if self.cuda_on:
                    dummy_tensor = dummy_tensor.cuda()
                vrd_detections.append({'relationships': dummy_tensor, 'relationships_scores': dummy_tensor,
                                       'attributes': dummy_tensor, 'attributes_scores': dummy_tensor})
                break

            limit = 80
            how_many = obj.size(0)
            if how_many > limit:
                obj = obj[:limit]
                l = l[:limit]
                targets[idx]['labels'] =  targets[idx]['labels'][:limit]
                targets[idx]['relationships'] = targets[idx]['relationships'][:limit, :limit]
                targets[idx]['attributes'] = targets[idx]['attributes'][:limit, :]
                print('Skipping... too many objects ({})'.format(how_many))

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
                inferred_attr = torch.sigmoid(attr_out)
                attr_scores, attr_indexes = torch.sort(inferred_attr, dim=1, descending=True)

            # Infer the relationships between objects

            # Pseudo-code:
            # for every couple:
                # compute the union bounding box
                # pool this region in order to extract features
                # concat subj+label, rel, obj+label features
                # pass through the relationships classifier

            # 0. Compute the union bounding box and the spatial features
            obj_perm = obj.unsqueeze(0).repeat(obj.size(0), 1, 1)  # K x K x 4
            subj_perm = obj.unsqueeze(1).repeat(1, obj.size(0), 1) # K x K x 4
            box1box2_perm = torch.cat((obj_perm, subj_perm), dim=2)  # K x K x 8
            relboxes = self.bbox_union(box1box2_perm, padding=10)  # K x K x 4
            box1relboxes_perm = torch.cat((obj_perm, relboxes), dim=2)  # K x K x 8
            relboxesbox2_perm = torch.cat((relboxes, subj_perm), dim=2)  # K x K x 8

            box1box2_perm_feats = self.spatial_features(box1box2_perm)  # K x K x 4
            so_feats = box1box2_perm_feats[:, :, :-2]   # K x K x 2, exclude areas
            area_boxes_over_img = box1box2_perm_feats[:, :, -2:] / (img.shape[-1] * img.shape[-2])   # take only the areas and normalize with respect to frame
            sp_feats = self.spatial_features(box1relboxes_perm)[:, :, :-2]  # K x K x 4
            po_feats = self.spatial_features(relboxesbox2_perm)[:, :, :-2]  # K x K x 4

            spatial_features = torch.cat([so_feats, sp_feats, po_feats, area_boxes_over_img], dim=2)
            '''
            for (idx1, box1), (idx2, box2) in itertools.permutations(enumerate(obj), 2):
                rel_box = self.bbox_union(box1, box2, padding=10)
                rel_boxes[idx1, idx2, :] = rel_box

                so = self.spatial_features(box1, box2)
                sp = self.spatial_features(box1, rel_box)
                po = self.spatial_features(rel_box, box2)
                area_boxes_over_img = torch.FloatTensor([(box1[2] - box1[0]) * (box1[3] - box1[1]) / (img.shape[-1] * img.shape[-2]),
                                                         (box2[2] - box2[0]) * (box2[3] - box2[1]) / (img.shape[-1] * img.shape[-2])])
                spatial_features[idx1, idx2] = torch.cat((so, sp, po, area_boxes_over_img))
            '''

            # 1. Compute the union bounding box, only if rel_context is not None
            if self.rel_context == 'relation_box':
                # 2. Pool all the regions
                relboxes = relboxes.view(-1, 4)
                pooled_rel_regions = torchvision.ops.roi_align(img_f.unsqueeze(0), [relboxes], output_size=(4, 4), spatial_scale=scale)   # K x K x 256 x 4 x 4
                # Prepare the relationship features for the concatenation
                pooled_rel_regions = pooled_rel_regions.view(obj.size(0), obj.size(0), -1)  # K x K x 256*4*4
            elif self.rel_context == 'whole_image':
                raise NotImplementedError()
                # TODO!
            elif self.rel_context == 'image_level_labels':
                raise NotImplementedError()
                # TODO!

            # Prepare the object features for concatenation
            pooled_obj_regions = pooled_regions.view(pooled_regions.size(0), -1).\
                unsqueeze(0).repeat(obj.size(0), 1, 1)   # K x K x 256*4*4
            pooled_subj_regions = pooled_regions.view(pooled_regions.size(0), -1).\
                unsqueeze(1).repeat(1, obj.size(0), 1)   # K x K x 256*4*4

            # Handle labels
            one_hot_obj_label = nn.functional.one_hot(l, self.num_classes).float().unsqueeze(0).repeat(obj.size(0), 1, 1) # K x K x num_classes
            one_hot_subj_label = nn.functional.one_hot(l, self.num_classes).float().unsqueeze(1).repeat(1, obj.size(0), 1) # K x K x num_classes

            # 3. Concatenate all the features
            pooled_concat = torch.cat(
                (
                    pooled_obj_regions, one_hot_obj_label,
                    pooled_subj_regions, one_hot_subj_label,
                    spatial_features
                ),
                dim=2
            )

            if self.rel_context is not None:
                # Add the information regarding the relationship context
                pooled_concat = torch.cat(
                    (
                        pooled_concat, pooled_rel_regions
                    ),
                    dim=2
                )


            if self.training:
                # If training, we suppress some of the relationships
                # Hence, calculate a filter in order to control the amount of relations and non-relations seen by the architecture.
                choosen_relation_indexes = self.choose_rel_indexes(targets[idx]['relationships'])
                pooled_concat = pooled_concat[choosen_relation_indexes]
            else:
                # Reshape for passing through the classifier
                pooled_concat = pooled_concat.view(obj.size(0) ** 2, -1)

            # 4. Run the Relationship classifier
            rel_out = self.relationships_classifier(pooled_concat)
            if self.training:
                t = targets[idx]['relationships'][choosen_relation_indexes]
                rel_loss += 10 * self.rel_loss_fn(rel_out, t)
            else:
                inferred_rels = F.softmax(rel_out, dim=1)
                rels_scores, rels_indexes = torch.max(inferred_rels, dim=1)

                # reshape back to a square matrix
                rels_scores = rels_scores.view(obj.size(0), obj.size(0))
                rels_indexes = rels_indexes.view(obj.size(0), obj.size(0))

            if not self.training:
                # accumulate VRD detections from every batch
                vrd_detections.append({'relationships': rels_indexes, 'relationships_scores':rels_scores,
                                       'attributes': attr_indexes, 'attributes_scores': attr_scores})

        if self.training:
            # Compute the mean losses over all detections for every batch
            num_objects_total = sum([t['boxes'].size(0) for t in targets])
            attr_loss /= num_objects_total
            rel_loss /= num_objects_total ** 2

            losses_dict.update({'relationships_loss': rel_loss, 'attributes_loss': attr_loss})
            return losses_dict

        else:
            # Merge boxes, inferred_attr and inferred_rels using the same interface of the detection in torchvision
            detections = [{**obj_det, **vrd_det} for obj_det, vrd_det in zip(detections, vrd_detections)]
            return detections

    def cuda(self, device=None):
        self.cuda_on = True
        return super().cuda(device)





