import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.transform import GeneralizedRCNNTransform

import pdb


class VRD(nn.Module):
    def __init__(self, detector, dataset):
        super(VRD, self).__init__()
        self.detector = detector.module if isinstance(detector, nn.DataParallel) else detector
        # we want the detector in evaluation mode
        self.detector.training = False
        self.training = True
        self.cuda_on = False
        self.num_classes = dataset.num_classes()
        self.num_attributes = dataset.num_attributes()
        self.num_relationships = dataset.num_relationships()

        self.attributes_classifier = nn.Sequential(
            nn.Linear(4 * 4 * 256 * 2 + self.num_classes, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.num_attributes),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.attr_loss_fn = nn.MultiLabelSoftMarginLoss(reduction='sum')

    def make_one_hot_labels(self, labels):
        '''
        Converts an integer label torch.autograd.Variable to a one-hot Variable.

        Parameters
        ----------
        labels : torch.autograd.Variable of torch.cuda.LongTensor
            Each value is an integer representing correct classification.

        Returns
        -------
        target : torch.autograd.Variable of torch.cuda.FloatTensor
            N x C, where C is class number. One-hot encoded.
        '''
        target = nn.functional.one_hot(labels, self.num_classes).type(torch.FloatTensor)
        if self.cuda_on:
            target = target.cuda()

        return target

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        # boxes_per_batch = [len(t['boxes']) for t in targets]

        original_image_sizes = [img.shape[-2:] for img in images]
        # transform images and targets
        images, targets = self.detector.transform(images, targets)

        if self.training:
            # objects from every batch
            objects = [t['boxes'] for t in targets]
            labels = [t['labels'] for t in targets]  # labels from every batch
        else:
            # forward through the object detector in order to retrieve objects from the image
            detections = self.detector(images)
            objects = [d['boxes'] for d in detections]
            labels = [d['labels'] for d in detections]

        # TODO: warning! Do we want gradient to be propagated into the backbone of the detector?
        image_features = self.detector.backbone(images.tensors)[3]
        image_features_pooled = self.avgpool(self.detector.backbone(images.tensors)['pool'])

        # iterate through batch size
        attr_loss = 0
        rel_loss = 0
        for idx, (img_f, img_f_pool, obj, l) in enumerate(zip(image_features, image_features_pooled, objects, labels)):
            pooled_regions = torchvision.ops.roi_align(img_f.unsqueeze(0), obj, output_size=(4, 4))

            # infer the attributes for every object in the images

            # 1. concatenate image_features, pooled_regions and labels
            attr_features = torch.cat(
                (
                pooled_regions.view(pooled_regions.size(0), -1),
                img_f_pool.view(-1).unsqueeze(0).expand(len(obj), -1),   # concatenate image level features to all the regions
                self.make_one_hot_labels(l)
                ),
                dim=1
            )
            pdb.set_trace()

            # 2. run the multi-label classifier
            attr_out = self.attributes_classifier(attr_features)
            if self.training:
                attr_loss += self.attr_loss_fn(attr_out, targets[idx]['attributes'].float())
            else:
                raise NotImplementedError()
                # TODO: infer attributes for every detection

            # infer the relationships between objects

            # for every couple:
                # compute the union bounding box
                # pool this region in order to extract features
                # concat subj+label, rel, obj+label features
                # pass through the relationships classifier

    def cuda(self, device=None):
        self.cuda_on = True
        return super().cuda(device)





