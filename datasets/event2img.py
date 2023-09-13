import copy

from PIL import Image

import torch
from torch.utils.data import Dataset

from .vis import events2frames
from .augment import RandAugment, InterpolationMode
from .utils import random_time_flip_events as tflip_events
from .utils import random_flip_events_along_x as hflip_events


class Event2ImageDataset(Dataset):
    """A wrapper for EventDataset that converts events to 2D images."""

    def __init__(
        self,
        transforms,
        event_dataset,
        quantize_args=dict(
            max_imgs=2,
            split_method='event_count',
            convert_method='event_histogram',
            N=30000,
            grayscale=True,
            count_non_zero=False,  # hotpixel statistics
            background_mask=True,  # apply white background via alpha-masking
        ),
        augment=False,
        tta=False,
    ):

        # data augmentation
        self.augment = augment
        if augment:
            self.augmentation = RandAugment(
                num_ops=2,  # follow common practice
                interpolation=InterpolationMode.BICUBIC,  # CLIP uses bicubic
                fill=[255, 255, 255]  # pad with white pixels
                if quantize_args['background_mask'] else [0, 0, 0],
            )

        # transforms to apply to the 2D images
        self.transforms = transforms

        # dataset that loads raw events in shape [N, 4 (x, y, t, p)]
        self.event_dataset = event_dataset
        self.classes = event_dataset.classes
        self.resolution = event_dataset.resolution
        self.max_t = event_dataset.max_t  # timestamp
        self.max_n = event_dataset.max_n  # number of events
        self.tta = tta
        if tta:
            assert not event_dataset.augmentation, \
                'Do not augment events in pseudo label generation'
            assert not augment, 'Do not augment twice'
            assert event_dataset.num_shots is None
            print('Apply h- and t-flip TTA in pseudo label generation')

        # arguments for mapping events to 2D images
        self.quantize_args = copy.deepcopy(quantize_args)
        self.quantize_args['shape'] = self.resolution

        self.split_method = quantize_args['split_method']
        self.event_rep = quantize_args['convert_method']
        assert self.split_method == 'event_count'
        max_imgs = round(self.max_n / quantize_args['N'])
        max_max_imgs = quantize_args.pop('max_imgs', 10)  # hard limit
        self.max_imgs = max(min(max_imgs, max_max_imgs), 1)

        # a hack in visualization to also load the raw events data
        self.keep_events = False

    def __len__(self):
        return len(self.event_dataset)

    def _subsample_imgs(self, imgs):
        """Randomly select a subset of images or pad with zeros."""
        valid_mask = torch.zeros(self.max_imgs).bool()
        if len(imgs) > self.max_imgs:
            valid_mask[:] = True
            idxs = torch.randperm(len(imgs))[:self.max_imgs]
            imgs = imgs[idxs]
        else:
            valid_mask[:len(imgs)] = True
            pad = torch.zeros(
                (self.max_imgs - len(imgs), *imgs.shape[1:])).type_as(imgs)
            imgs = torch.cat([imgs, pad], dim=0)
        return imgs, valid_mask

    def _load_tta_data(self, idx):
        """Apply h- and t-flip to the loaded events, then convert to images."""
        data_dict = self.event_dataset[idx]
        events = data_dict.pop('events')
        assert not self.keep_events, 'val dataset should not be TTA'
        h_events = hflip_events(
            copy.deepcopy(events), resolution=self.resolution, p=1.)
        t_events = tflip_events(copy.deepcopy(events), p=1.)
        h_t_events = tflip_events(copy.deepcopy(h_events), p=1.)
        tta_events = [events, h_events, t_events, h_t_events]
        tta_imgs, tta_valid_mask = [], []
        for events in tta_events:
            imgs, valid_mask = self._event2img(events)
            tta_imgs.append(imgs)
            tta_valid_mask.append(valid_mask)
        data_dict['img'] = torch.stack(tta_imgs, dim=0)  # [4, N, 3, H, W]
        data_dict['valid_mask'] = torch.stack(tta_valid_mask, dim=0)  # [4, N]
        # `label` is still just an integer
        return data_dict

    def _event2img(self, events):
        """Convert events to 2D images."""
        # events: [N, 4 (x, y, t, p)]
        # get [N, H, W, 3] images with dtype np.uint8
        imgs = events2frames(events, **self.quantize_args)
        imgs = [Image.fromarray(img) for img in imgs]
        if self.augment:
            imgs = self.augmentation(imgs)
        imgs = torch.stack([self.transforms(img) for img in imgs])
        # to [N, 3, H, W] torch.Tensor as model inputs

        # randomly select a subset of images or pad with zeros
        imgs, valid_mask = self._subsample_imgs(imgs)

        return imgs, valid_mask

    def __getitem__(self, idx):
        if self.tta:
            return self._load_tta_data(idx)

        data_dict = self.event_dataset[idx]
        events = data_dict.pop('events')

        if self.keep_events:
            data_dict['events'] = copy.deepcopy(events)

        imgs, valid_mask = self._event2img(events)

        data_dict['img'] = imgs
        data_dict['valid_mask'] = valid_mask

        return data_dict


def build_event2img_dataset(params, event_dataset, augment=False, tta=False):
    """Wrap an event dataset with a Event2Image processing pipeline."""
    return Event2ImageDataset(
        transforms=params.data_transforms,
        event_dataset=event_dataset,
        quantize_args=params.quantize_args,
        augment=augment,
        tta=tta,
    )
