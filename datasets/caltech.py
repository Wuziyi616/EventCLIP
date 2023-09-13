import os
from os import listdir
from os.path import join
import random

import numpy as np

from torch.utils.data import Dataset

from nerv.utils import load_obj, dump_obj

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


def get_real_path(path):
    while os.path.islink(path):
        path = os.readlink(path)
    return path


class NCaltech101(Dataset):
    """Dataset class for N-Caltech101 dataset."""

    def __init__(
        self,
        root,
        augmentation=False,
        num_shots=None,
        repeat=True,
        new_cnames=None,
    ):
        self.root = root
        self.classes = sorted(listdir(root))
        # TODO: a hack for identifying generated pseudo labeled datasets
        self.is_pseudo = 'pseudo' in root
        if self.is_pseudo:
            print('Using pseudo labeled dataset!')

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
        self.few_shot = (num_shots is not None and num_shots > 0)
        if self.few_shot:
            assert 'train' in root.lower(), 'Only sample data in training set'
        self.repeat = repeat

        self.labeled_files, self.labels = self._get_sample_idx()
        assert len(self.labeled_files) == len(self.labels)

        # change some class names
        self.new_cnames = new_cnames
        if new_cnames is None:
            return
        for i in range(len(self.classes)):
            if self.classes[i] in new_cnames:
                new_name = new_cnames[self.classes[i]]
                print(f'Rename {self.classes[i]} to {new_name}')
                self.classes[i] = new_name

    def _get_sample_idx(self):
        """Load event file_name and label pairs."""
        # load pre-generated splits if available
        if self.few_shot and not self.is_pseudo:
            cur_dir = os.path.dirname(os.path.realpath(__file__))
            split_fn = os.path.join(
                cur_dir, 'files', self.__class__.__name__,
                f'{self.num_shots}shot-repeat={self.repeat}.pkl')
            if os.path.exists(split_fn):
                print(f'Loading pre-generated split from {split_fn}')
                splits = load_obj(split_fn)  # Dict[event_fn: label]
                labeled_files = np.array(list(splits.keys()))
                labels = np.array(list(splits.values()))
                return labeled_files, labels

        labeled_files, labels = [], []

        # fix the random seed since we'll sample data
        random.seed(0)
        for i, c in enumerate(self.classes):
            cls_files = [
                get_real_path(join(self.root, c, f))
                for f in sorted(listdir(join(self.root, c)))
            ]
            if len(cls_files) == 0:
                print(f'Warning: class {c} has no data!')
                continue

            # randomly sample `num_shots` labeled data for each class
            if self.few_shot:
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
                raise ValueError(f'Invalid num_shots: {self.num_shots}')
            labeled_files += lbl_files
            labels += [i] * len(lbl_files)

        # save the splits for future use
        if self.few_shot and not self.is_pseudo:
            splits = {fn: lbl for fn, lbl in zip(labeled_files, labels)}
            os.makedirs(os.path.dirname(split_fn), exist_ok=True)
            dump_obj(splits, split_fn)
            print(f'Saving split file to {split_fn}')

        labeled_files = np.array(labeled_files)
        labels = np.array(labels)
        return labeled_files, labels

    def __len__(self):
        return len(self.labeled_files)

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
        f = str(self.labeled_files[idx])
        label = int(self.labels[idx])
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
        new_cnames=NEW_CNAMES,
    )
    val_set = NCaltech101(
        root=os.path.join(params.data_root, 'testing'),
        augmentation=False,
        new_cnames=NEW_CNAMES,
    )
    return train_set, val_set
