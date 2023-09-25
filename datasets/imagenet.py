import os

import numpy as np

from .caltech import NCaltech101


def load_event(event_path):
    """Load event data from npz file."""
    event = np.load(event_path)['event_data']
    event = np.stack([
        event['x'],
        event['y'],
        event['t'],
        event['p'].astype(np.uint8),
    ], 1)  # [N, 4]

    event = event.astype(float)

    # Account for int-type timestamp
    event[:, 2] /= 1e6

    # Account for zero polarity
    if event[:, 3].min() >= -0.5:
        event[:, 3][event[:, 3] <= 0.5] = -1

    return event


class NImageNet(NCaltech101):
    """Dataset class for N-ImageNet dataset."""

    def __init__(
        self,
        root,
        augmentation=False,
        num_shots=None,
    ):
        super().__init__(
            root=root,
            augmentation=augmentation,
            num_shots=num_shots,
            repeat=False,
            new_cnames=None,
        )

        # data stats
        self.resolution = (480, 640)
        self.max_t = 0.055  # max
        self.max_n = 135000  # 95th percentile

        # data augmentation
        self.flip_time = True

        # load folder name to class name mapping
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        label_map = os.path.join(cur_dir, 'files/CLIP-IN_ClassNames.txt')
        with open(label_map, 'r') as f:
            lines = f.readlines()[:1000]
            lines = [line.strip() for line in lines]
            """
            n01440764 tench
            n01443537 goldfish
            n01484850 great white shark
            n01491361 tiger shark
            n01494475 hammerhead shark
            """
        folder2name = {
            s.split(' ')[0]: ' '.join(s.split(' ')[1:])
            for s in lines
        }
        self.folder2name = folder2name
        self.name2folder = {v: k for k, v in folder2name.items()}
        self.classes = [folder2name[c] for c in self.classes]

    @staticmethod
    def _load_events(event_path):
        """Load events from a file."""
        return load_event(event_path).astype(np.float32)


def build_n_imagenet_dataset(
    params,
    val_only=False,
    gen_data=False,
    subset=-1,
):
    """Build the N-ImageNet dataset."""
    val_names = {
        1: 'val_mode_1',
        2: 'val_mode_5',
        3: 'val_mode_6',
        4: 'val_mode_7',
        5: 'val_mode_3',
        6: 'val_brightness_4',
        7: 'val_brightness_5',
        8: 'val_brightness_6',
        9: 'val_brightness_7',
    }
    if subset > 0:
        val_set = val_names[subset]
        val_root = os.path.join(params.data_root, f'extracted_{val_set}')
        print('Using N-ImageNet subset:', val_set)
    else:
        val_root = os.path.join(params.data_root, 'extracted_val')
        print('Using normal N-ImageNet val set')

    # only build the test set
    test_set = NImageNet(
        root=val_root,
        augmentation=False,
    )
    if val_only:
        assert not gen_data
        return test_set
    # build the training set for pseudo label generation
    if gen_data:
        return NImageNet(
            root=os.path.join(params.data_root, 'extracted_train'),
            augmentation=False,
        )

    # build the training set
    train_set = NImageNet(
        root=os.path.join(params.data_root, 'extracted_train'),
        augmentation=True,
        num_shots=params.get('num_shots', None),
    )
    return train_set, test_set
