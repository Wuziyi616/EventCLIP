import os
from os import listdir
from os.path import join
import random

import numpy as np

from torch.utils.data import Dataset

from .utils import random_time_flip_events, random_shift_events, \
    random_flip_events_along_x, center_events


class NCaltech101(Dataset):
    """Dataset class for N-Caltech101 dataset."""

    def __init__(self, root, augmentation=False, num_shots=None, repeat=True):
        self.root = root
        self.classes = sorted(listdir(root))

        # data stats (computed from the test set)
        self.resolution = (180, 240)
        # t is very uniform, i.e. different samples have similar max_t
        # so just take the max (unit: second)
        self.max_t = 0.325
        # the number of events are VERY unbalanced, so instead of taking
        # the max, we take the 95th percentile
        self.max_n = 225000

        # data augmentation
        self.augmentation = augmentation
        self.flip_time = False
        self.max_shift = 20

        # few-shot cls
        self.num_shots = num_shots
        self.repeat = repeat

        self.files, self.labels = self._get_sample_idx()

    def _get_sample_idx(self):
        """Load event file_name and label pairs."""
        files, labels = [], []

        # fix the random seed since we'll sample data
        random.seed(0)
        for i, c in enumerate(self.classes):
            new_files = [
                join(self.root, c, f) for f in sorted(listdir(join(self.root, c)))
            ]

            # randomly sample `num_shots` data for each class
            if self.num_shots is not None:
                if self.num_shots <= len(new_files):
                    new_files = random.sample(new_files, k=self.num_shots)
                else:
                    if self.repeat:
                        new_files = random.choices(new_files, k=self.num_shots)
                    else:
                        pass

            files += new_files
            labels += [i] * len(new_files)

        files = np.array(files)
        labels = np.array(labels)
        return files, labels

    def __len__(self):
        return len(self.files)

    def _rand_another(self):
        """Randomly sample another data."""
        idx = np.random.randint(0, len(self))
        return self.__getitem__(idx)

    @staticmethod
    def _load_events(event_path):
        """Load events from a file."""
        return np.load(event_path).astype(np.float32)

    def _augment_events(self, events):
        """Data augmentation on events."""
        if self.flip_time:
            events = random_time_flip_events(events)
        events = random_shift_events(
            events, max_shift=self.max_shift, resolution=self.resolution)
        events = random_flip_events_along_x(events, resolution=self.resolution)
        # not using time flip on N-Caltech and N-Cars dataset
        return events

    def __getitem__(self, idx):
        """
        returns events and label, potentially with augmentation
        :param idx: data_idx
        :return: [N, (x,y,t,p)], label, data_idx
        """
        label = int(self.labels[idx])
        f = str(self.files[idx])
        events = self._load_events(f)
        # the spatial resolution of N-Caltech events is 180x240
        # we should center the spatial coordinates of events
        # some events only reside in e.g. [0, 0] x [100, 160]
        # which will be largely removed after center crop!
        events = center_events(events, resolution=self.resolution)

        if self.augmentation:
            events = self._augment_events(events)

        if events.shape[0] == 0:
            return self._rand_another()

        # events: [N, 4 (x, y, t, p)], label: int
        # N is usually 1e5 ~ 1e6
        return {
            'events': events,
            # 't': events[:, 2],
            'label': label,
            'data_idx': idx,
        }


def build_n_caltech_dataset(params, val_only=False):
    """Build the N-Caltech101 dataset."""
    # only build the test set
    if val_only:
        return NCaltech101(
            root=os.path.join(params.data_root, 'testing'),
            augmentation=False,
            num_shots=None,
        )

    # build the training set
    train_set = NCaltech101(
        root=os.path.join(params.data_root, 'training'),
        augmentation=True,
        num_shots=params.get('num_shots', None),
        repeat=params.get('repeat_data', True),
    )
    val_set = NCaltech101(
        root=os.path.join(params.data_root, 'testing'),
        augmentation=False,
        num_shots=None,
    )
    return train_set, val_set
