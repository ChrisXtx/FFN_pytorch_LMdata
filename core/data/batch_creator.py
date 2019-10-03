from torch.utils import data
from typing import Tuple, Dict, Optional, Union, Sequence, Any, List, Callable
import logging
import numpy as np
import torch
from scipy.special import logit
from .utils import *
import random
import itertools
from torch.autograd import Variable
import cv2


class BatchCreator(data.Dataset):
    def __init__(
            self,
            input_h5data: List[str],
            input_size: Sequence[int],
            delta: Sequence[int],
            threshold: float = 0.9,
            train: bool = True):
        super(BatchCreator, self).__init__()

        self.shifts = []
        self.deltas = delta
        for dx in (-self.deltas[0], 0, self.deltas[0]):
            for dy in (-self.deltas[1], 0, self.deltas[1]):
                for dz in (-self.deltas[2], 0, self.deltas[2]):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    self.shifts.append((dx, dy, dz))
        random.shuffle(self.shifts)

        self.input_data = []
        self.label_data = []
        self.seed = []
        self.coor = []
        self.parse_h5py(input_h5data)

        # Early checks
        if len(self.input_data) != len(self.label_data):
            raise ValueError("input_h5data and target_h5data must be lists of same length!")
        if not train:
            logger.warning('Augmentations should not be used on validation data.')

        self.input_size = np.array(input_size)
        self.seed_shape = self.input_size + np.array(delta) * 2
        self.label_radii = self.seed_shape // 2

        self.data_idx = 0
        self.coor_idx = 0

        self.image_patch = None
        self.label_patch = None
        self.seed_patch = None
        self.offset = None

        self.threshold = threshold
        self.num_classes = 2
        self.reset = True

    def parse_h5py(self, input):
        for input_data in input:
            with h5py.File(input_data, 'r') as raw:
                self.input_data.append(raw['image'].value)
                self.label_data.append(raw['label'].value)
                self.coor.append(raw['coor'].value)
                self.seed.append(logit(np.full(list(raw['image'].value.shape), 0.05, dtype=np.float32)))

    def __getitem__(self, idx):
        self.coor_patch = self.coor[self.data_idx][idx]

        start = self.coor_patch - self.seed_shape // 2
        end = start + self.seed_shape

        assert np.all(start >= 0)

        selector = [slice(s, e) for s, e in zip(start, end)]
        self.image_patch = self.input_data[self.data_idx][selector]
        self.label_patch = self.label_data[self.data_idx][selector]
        self.label_patch = np.logical_and(self.label_patch > 0, np.equal(self.label_patch, self.label_patch[tuple(self.label_radii)]))
        self.label_patch = np.where(self.label_patch, np.ones(self.label_patch.shape)*0.95, np.ones(self.label_patch.shape)*0.05)
        self.seed_patch = self.seed[self.data_idx][selector]
        self.seed_patch[tuple(self.label_radii)] = logit(0.95)

        return torch.from_numpy(self.image_patch).float(), \
               torch.from_numpy(self.label_patch),\
               torch.from_numpy(self.seed_patch), self.coor_patch

    def __len__(self):
        return len(self.coor[self.data_idx])
