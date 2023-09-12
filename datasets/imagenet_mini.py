import os

from .imagenet import NImageNet

# N-ImageNet (Mini) subset, taken from https://arxiv.org/pdf/2308.09383.pdf
# Note that this is slightly different from the Mini-ImageNet used in e.g. MAML
MINI_NAMES = [
    "hamster", "academic gown", "airship", "jackfruit", "barbershop",
    "cocktail shaker", "Komodo dragon", "sunglasses", "grey fox", "cello",
    "comic book", "goldfish", "Bloodhound", "porcupine", "jaguar", "kingsnake",
    "altar", "water buffalo", "chiton", "scarf", "storage chest", "tool kit",
    "sea anemone", "Border Terrier", "menu", "picket fence", "forklift",
    "yellow lady's slipper", "chameleon", "dragonfly", "Pomeranian",
    "European garden spider", "Airedale Terrier", "frilled-necked lizard",
    "black stork", "valley", "radio telescope", "leopard", "crossword",
    "Australian Terrier", "Shih Tzu", "husky", "can opener", "artichoke",
    "assault rifle", "fountain pen", "harvestman", "parallel bars",
    "harmonica", "half-track", "snoek fish", "pencil sharpener", "submarine",
    "muzzle", "eastern diamondback rattlesnake", "Miniature Schnauzer",
    "missile", "Komondor", "grand piano", "website", "king penguin", "canoe",
    "red-breasted merganser", "trolleybus", "quail", "poke bonnet",
    "King Charles Spaniel", "race car", "Malinois", "solar thermal collector",
    "slug", "bucket", "dung beetle", "Asian elephant", "window screen",
    "Flat-Coated Retriever", "steel drum", "snowplow", "handkerchief",
    "tailed frog", "church", "Chesapeake Bay Retriever", "Christmas stocking",
    "hatchet", "hair clip", "vulture", "sidewinder rattlesnake",
    "oscilloscope", "worm snake", "eel", "wok", "planetarium",
    "Old English Sheepdog", "platypus", "Pembroke Welsh Corgi",
    "alligator lizard", "consomme", "African rock python", "hot tub",
    "Tibetan Mastiff"
]


class NImageNetMini(NImageNet):
    """Dataset class for N-ImageNet (Mini) dataset."""

    def __init__(
        self,
        root,
        augmentation=False,
        num_shots=None,
        semi_shots=None,
    ):
        self.root = root

        # data stats
        self.resolution = (480, 640)
        self.max_t = 0.055  # max
        self.max_n = 135000  # 95th percentile

        # data augmentation
        self.augmentation = augmentation
        self.flip_time = True
        self.max_shift = 20

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
        # only take a subset of 100 classes
        folder2name = {k: v for k, v in folder2name.items() if v in MINI_NAMES}
        assert len(folder2name) == 100 == len(MINI_NAMES)
        self.classes = list(folder2name.keys())
        self.folder2name = folder2name
        self.name2folder = {v: k for k, v in folder2name.items()}

        # few-shot cls
        self.num_shots = num_shots  # number of labeled data per class
        self.semi_shots = semi_shots  # number of unlabeled data per class
        self.un_sup = (num_shots is not None and num_shots == 0)
        self.semi_sup = (semi_shots is not None and semi_shots > 0)

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

        # finally, get semantically meaningful class names
        self.classes = [folder2name[c] for c in self.classes]
        assert all(c in self.classes for c in MINI_NAMES) and \
            len(self.classes) == 100


def build_n_imagenet_mini_dataset(params, val_only=False, gen_data=False):
    """Build the N-ImageNet (Mini) dataset."""
    val_root = os.path.join(params.data_root, 'extracted_val')

    # only build the test set
    test_set = NImageNetMini(
        root=val_root,
        augmentation=False,
    )
    if val_only:
        assert not gen_data, 'Only generate pseudo labels on the training set'
        return test_set
    # build the training set for pseudo label generation
    if gen_data:
        return NImageNetMini(
            root=os.path.join(params.data_root, 'extracted_train'),
            augmentation=False,
        )

    # build the training set
    train_set = NImageNetMini(
        root=os.path.join(params.data_root, 'extracted_train'),
        augmentation=True,
        num_shots=params.get('num_shots', None),
        semi_shots=params.get('semi_shots', None),
    )
    return train_set, test_set
