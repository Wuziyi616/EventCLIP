import os
from os import listdir
from os.path import join
import random

import numpy as np

from torch.utils.data import Dataset

from .utils import random_time_flip_events, random_shift_events, \
    random_flip_events_along_x, center_events

# from https://github.com/KaiyangZhou/CoOp/blob/main/datasets/caltech101.py
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",  # actually doesn't exist
    "Faces_easy": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
    "BACKGROUND_Google": "background",  # random images, hard to categorize
}


class NCaltech101(Dataset):
    """Dataset class for N-Caltech101 dataset."""

    def __init__(
        self,
        root,
        augmentation=False,
        num_shots=None,
        repeat=True,
        semi_shots=None,
        new_cnames=None,
    ):
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
        self.num_shots = num_shots  # number of labeled data per class
        self.semi_shots = semi_shots  # number of unlabeled data per class
        self.un_sup = (num_shots is not None and num_shots == 0)
        self.semi_sup = (semi_shots is not None and semi_shots > 0)
        self.repeat = repeat

        self.labeled_files, self.unlabeled_files, self.labels, self.un_labels = \
            self._get_sample_idx()
        assert len(self.labeled_files) == len(self.labels)
        assert len(self.unlabeled_files) == len(self.un_labels)
        if self.un_sup:
            assert len(self.labeled_files) == 0
            assert len(self.unlabeled_files) > 0
            print(f'\nUnsupervised learning with {self.semi_shots=}\n')
        if self.semi_sup:
            # assert len(self.labeled_files) > 0
            assert len(self.unlabeled_files) > 0
            print(f'\nSemi-supervised learning with {self.num_shots=} '
                  f'and {self.semi_shots=}\n')

        # change some class names
        if new_cnames is None:
            return
        for i in range(len(self.classes)):
            if self.classes[i] in new_cnames:
                new_name = new_cnames[self.classes[i]]
                print(f'Rename {self.classes[i]} to {new_name}')
                self.classes[i] = new_name

    def _get_sample_idx(self):
        """Load event file_name and label pairs."""
        labeled_files, unlabeled_files, labels, un_labels = [], [], [], []

        # fix the random seed since we'll sample data
        random.seed(0)
        for i, c in enumerate(self.classes):
            cls_files = [
                join(self.root, c, f)
                for f in sorted(listdir(join(self.root, c)))
            ]
            if len(cls_files) == 0:
                print(f'Warning: class {c} has no data!')
                continue

            # randomly sample `num_shots` labeled data for each class
            if self.num_shots is not None and self.num_shots > 0:
                if self.num_shots <= len(cls_files):
                    lbl_files = random.sample(cls_files, k=self.num_shots)
                else:
                    if self.repeat:
                        lbl_files = random.choices(cls_files, k=self.num_shots)
                    else:
                        lbl_files = cls_files
            elif self.num_shots is None:
                lbl_files = cls_files
            else:
                assert self.un_sup
                lbl_files = []
            labeled_files += lbl_files
            labels += [i] * len(lbl_files)

            # randomly sample `semi_shots` unlabeled data for each class
            if not self.semi_sup:
                continue
            cls_files = sorted(list(set(cls_files) - set(lbl_files)))
            if len(cls_files) == 0:
                continue
            if self.semi_shots <= len(cls_files):
                unlbl_files = random.sample(cls_files, k=self.semi_shots)
            else:  # don't repeat data
                unlbl_files = cls_files
            # no overlap
            assert len(set(lbl_files) & set(unlbl_files)) == 0
            unlabeled_files += unlbl_files
            un_labels += [i] * len(unlbl_files)

        labeled_files = np.array(labeled_files)
        unlabeled_files = np.array(unlabeled_files)
        labels = np.array(labels)
        un_labels = np.array(un_labels)
        return labeled_files, unlabeled_files, labels, un_labels

    def __len__(self):
        return len(self.labeled_files) + len(self.unlabeled_files)

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
        if idx < len(self.labeled_files):
            f = str(self.labeled_files[idx])
            label = int(self.labels[idx])
        else:
            f = str(self.unlabeled_files[idx - len(self.labeled_files)])
            label = -1 * int(self.un_labels[idx - len(self.labeled_files)]) - 1
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


def build_n_caltech_dataset(params, val_only=False, gen_data=False):
    """Build the N-Caltech101 dataset."""
    # only build the test set
    if val_only:
        assert not gen_data, 'Only generate pseudo labels on the training set'
        return NCaltech101(
            root=os.path.join(params.data_root, 'testing'),
            augmentation=False,
            new_cnames=NEW_CNAMES,
        )
    # build the training set for pseudo label generation
    if gen_data:
        return NCaltech101(
            root=os.path.join(params.data_root, 'training'),
            augmentation=False,
            new_cnames=NEW_CNAMES,
        )

    # build the training set
    train_set = NCaltech101(
        root=os.path.join(params.data_root, 'training'),
        augmentation=True,
        num_shots=params.get('num_shots', None),
        repeat=params.get('repeat_data', True),
        semi_shots=params.get('semi_shots', None),
        new_cnames=NEW_CNAMES,
    )
    val_set = NCaltech101(
        root=os.path.join(params.data_root, 'testing'),
        augmentation=False,
        new_cnames=NEW_CNAMES,
    )
    return train_set, val_set
