import copy

from PIL import Image

import torch
from torch.utils.data import Dataset

from .vis import events2frames
from .augment import RandAugment, InterpolationMode


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

        # semi-supervised learning
        self.semi_sup = event_dataset.semi_sup

    def __len__(self):
        return len(self.event_dataset)

    def _subsample_imgs(self, imgs):
        # randomly select a subset of images or pad with zeros
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

    def _load_semi_sup_data(self, idx):
        """Apply a strong data_aug and a weak data_aug to the same image."""
        data_dict = self.event_dataset[idx]
        events = data_dict.pop('events')
        assert not self.keep_events, 'val dataset should not be semi-sup'

        # get [N, H, W, 3] images with dtype np.uint8
        imgs = events2frames(events, **self.quantize_args)
        imgs = [Image.fromarray(img) for img in imgs]
        weak_imgs = torch.stack([self.transforms(img) for img in imgs]).clone()
        if self.augment:
            imgs = self.augmentation(imgs)
        strong_imgs = torch.stack([self.transforms(img) for img in imgs])
        # to [N, 3, H, W] torch.Tensor as model inputs

        # randomly select a subset of images or pad with zeros
        weak_imgs, valid_mask = self._subsample_imgs(weak_imgs)
        strong_imgs, valid_mask_ = self._subsample_imgs(strong_imgs)
        assert (valid_mask == valid_mask_).all()

        data_dict['weak_img'] = weak_imgs
        data_dict['strong_img'] = strong_imgs
        data_dict['valid_mask'] = valid_mask

        return data_dict

    def __getitem__(self, idx):
        if self.semi_sup:
            return self._load_semi_sup_data(idx)

        data_dict = self.event_dataset[idx]
        events = data_dict.pop('events')

        if self.keep_events:
            data_dict['events'] = copy.deepcopy(events)

        # get [N, H, W, 3] images with dtype np.uint8
        imgs = events2frames(events, **self.quantize_args)
        imgs = [Image.fromarray(img) for img in imgs]
        if self.augment:
            imgs = self.augmentation(imgs)
        imgs = torch.stack([self.transforms(img) for img in imgs])
        # to [N, 3, H, W] torch.Tensor as model inputs

        # randomly select a subset of images or pad with zeros
        imgs, valid_mask = self._subsample_imgs(imgs)

        data_dict['img'] = imgs
        data_dict['valid_mask'] = valid_mask

        return data_dict


def build_event2img_dataset(params, event_dataset, augment=False):
    """Wrap an event dataset with a Event2Image processing pipeline."""
    return Event2ImageDataset(
        transforms=params.data_transforms,
        event_dataset=event_dataset,
        quantize_args=params.quantize_args,
        augment=augment,
    )
