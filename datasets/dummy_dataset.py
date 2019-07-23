import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class DummyDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

    def __getitem__(self, idx):
        random_size = torch.randint(low=1, high=1000, size=(2,))
        img = torch.rand(3, random_size[0], random_size[1])
        img = F.to_pil_image(img)
        # sample = {'img': img, 'annot': annot}

        target = {}
        target['boxes'] = torch.rand(10, 4) * 100
        target['boxes'][:, 2:4] = target['boxes'][:, :2] + torch.rand(1) * 80  # keep x2,y2 greater than x1,y1
        target['labels'] = torch.randint(low=1, high=600, size=(10,))
        if self.transform:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return 1000000

    def image_aspect_ratio(self, image_index):
        return 1

    def num_classes(self):
        return 600
