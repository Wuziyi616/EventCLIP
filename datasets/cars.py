import os

from .caltech import NCaltech101


class NCars(NCaltech101):
    """Dataset class for N-Cars dataset."""

    def __init__(self, root, augmentation=False, num_shots=None, repeat=True):
        super().__init__(root, augmentation, num_shots, repeat)

        # data stats
        self.resolution = (100, 120)
        self.max_t = 0.1  # max
        self.max_n = 12500  # 95th percentile

        # data augmentation
        self.max_shift = 10  # resolution is ~half as N-Caltech101

        # we probably want to change the class names
        # 'cars' --> 'car'
        # 'background' --> 'no car'?
        for i in range(len(self.classes)):
            if self.classes[i] == 'cars':
                self.classes[i] = 'car'
            # elif self.classes[i] == 'background':
            #     self.classes[i] = 'no car'


def build_n_cars_dataset(params, val_only=False):
    """Build the N-Cars dataset."""
    # only build the test set
    test_set = NCars(
        root=os.path.join(params.data_root, 'test'),
        augmentation=False,
        num_shots=None,
    )
    if val_only:
        return test_set

    # build the training set
    train_set = NCars(
        root=os.path.join(params.data_root, 'train'),
        augmentation=True,
        num_shots=params.get('num_shots', None),
        repeat=False,
    )
    return train_set, test_set
