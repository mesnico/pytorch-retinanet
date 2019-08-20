import random
import torch
import numpy as np

from torchvision.transforms import functional as F
from torchvision.models.detection.transform import resize_boxes
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        out_target = {
            'boxes': torch.as_tensor(target['boxes'], dtype=torch.float32),
            'labels': torch.as_tensor(target['labels'], dtype=torch.int64)
        }

        # if present, transforms relationships and attributes in sparse pytorch representations
        if 'relationships' in target:
            '''coo = target['relationships'].tocoo()
            sparse_rel = torch.sparse.LongTensor(torch.LongTensor([coo.row.tolist(), coo.col.tolist()]),
                                                 torch.LongTensor(coo.data.astype(np.int32)))
            '''
            rel_tensor = torch.LongTensor(target['relationships'].todense())
            out_target['relationships'] = rel_tensor

        if 'attributes' in target:
            '''coo = target['attributes'].tocoo()
            sparse_rel = torch.sparse.LongTensor(torch.LongTensor([coo.row.tolist(), coo.col.tolist()]),
                                                 torch.LongTensor(coo.data.astype(np.int32)))
            '''
            attr_tensor = torch.LongTensor(target['attributes'].todense())
            out_target['attributes'] = attr_tensor

        return image, out_target


class Resizer(object):
    """Convert ndarrays in sample to Tensors.

    NOTE: torchvision 0.3 already comes with a resizer, but it is embedded into the model. This obliges the model to fully
    load every image into GPU. Instead, this transformer pre-processes the image resizing it before loading onto
    the GPU.
    """

    def __init__(self, min_side=800, max_side=1333):
        self.min_side = min_side
        self.max_side = max_side

    def __call__(self, image, target):
        h, w = image.shape[-2:]
        min_size = float(min(image.shape[-2:]))
        max_size = float(max(image.shape[-2:]))

        size = self.min_side
        scale_factor = size / min_size
        if max_size * scale_factor > self.max_side:
            scale_factor = self.max_side / max_size
        image = torch.nn.functional.interpolate(
            image[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        return image, target


class Augment(object):
    def __init__(self):
        self.seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontal flips
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(0.5,
                          iaa.GaussianBlur(sigma=(0, 0.5))
                          ),
            # Strengthen or weaken the contrast in each image.
            iaa.ContrastNormalization((0.75, 1.5)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.Sometimes(0.5,
                          iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.15), per_channel=0.5)
                          ),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            iaa.Affine(
                rotate=(-20, 20)
            )
        ], random_order=True)  # apply augmenters in random order

    def __call__(self, image, target):
        np_image = image.permute(1, 2, 0).numpy()
        bboxes = [BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2) for x1, y1, x2, y2 in target['boxes'][:, :4]]
        image_aug, boxes_aug = self.seq(image=np_image, bounding_boxes=bboxes)

        image_aug = torch.from_numpy(np.ascontiguousarray(image_aug)).permute(2, 0, 1)
        boxes_aug = [[b.x1, b.y1, b.x2, b.y2] for b in boxes_aug]
        target['boxes'][:, :4] = torch.tensor(boxes_aug)

        return image_aug, target
