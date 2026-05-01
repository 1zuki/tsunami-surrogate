from __future__ import annotations

import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class ToTensor:
    def __call__(self, sample):
        sample = dict(sample)
        sample['x'] = torch.as_tensor(sample['x']).float()
        sample['y'] = torch.as_tensor(sample['y']).float()
        return sample
