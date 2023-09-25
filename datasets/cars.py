import os

from .caltech import NCaltech101

NEW_CNAMES = {
    "cars": "car",
    "background": "background",
}


class NCars(NCaltech101):
    """Dataset class for N-Cars dataset."""

    def __init__(
        self,
        root,
        augmentation=False,
        num_shots=None,
        new_cnames=None,
    ):
        super().__init__(
            root=root,
            augmentation=augmentation,
            num_shots=num_shots,
            repeat=False,
            new_cnames=new_cnames,
        )

        # data stats
        self.resolution = (100, 120)
        self.max_t = 0.1  # max
        self.max_n = 12500  # 95th percentile

        # data augmentation
        self.max_shift = 10  # resolution is ~half as N-Caltech101


def build_n_cars_dataset(params, val_only=False, gen_data=False):
    """Build the N-Cars dataset."""
    # only build the test set
    test_set = NCars(
        root=os.path.join(params.data_root, 'test'),
        augmentation=False,
        new_cnames=NEW_CNAMES,
    )
    if val_only:
        assert not gen_data
        return test_set
    # build the training set for pseudo label generation
    if gen_data:
        return NCars(
            root=os.path.join(params.data_root, 'train'),
            augmentation=False,
            new_cnames=NEW_CNAMES,
        )

    # build the training set
    train_set = NCars(
        root=os.path.join(params.data_root, 'train'),
        augmentation=True,
        num_shots=params.get('num_shots', None),
        new_cnames=NEW_CNAMES,
    )
    return train_set, test_set
