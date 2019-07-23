import random
import torch

from torchvision.transforms import functional as F
from torchvision.models.detection.transform import resize_boxes


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
        target = {
            'boxes': torch.as_tensor(target['boxes'], dtype=torch.float32),
            'labels': torch.as_tensor(target['labels'], dtype=torch.int64)
        }
        return image, target


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
