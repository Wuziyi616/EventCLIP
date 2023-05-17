import copy
import wandb
import numpy as np

import torch

from nerv.training import BaseMethod, CosineAnnealingWarmupRestarts

from datasets import events2frames


def denormalize(x):
    """Reverse the input image normalization."""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).type_as(x)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).type_as(x)
    return x * std[None, :, None, None] + mean[None, :, None, None]


def build_method(**kwargs):
    params = kwargs['params']
    if params.model in ['ZSCLIP', 'FSCLIP']:
        return EventCLIPMethod(**kwargs)
    else:
        raise NotImplementedError(f'{params.model} method is not implemented.')


class EventBaseMethod(BaseMethod):
    """Base method in this project."""

    def _round(self, v, n):
        if isinstance(v, (float, int)):
            return float(round(v, n))
        return type(v)([self._round(i, n) for i in v])

    @staticmethod
    def _convert_video(video, caption=None):
        """Convert torch.FloatTensor video to wandb.Video."""
        assert isinstance(video, torch.Tensor)
        video = denormalize(video)
        video = (video * 255.).cpu().numpy()
        video = np.round(video).clip(0, 255).astype(np.uint8)
        return wandb.Video(video, fps=2, caption=caption)

    @staticmethod
    def _get_sample_idx(N, dst):
        """Load data uniformly from the dataset."""
        dst_len = len(dst)
        N = N - 1 if dst_len % N != 0 else N
        sampled_idx = torch.arange(0, dst_len, dst_len // N)
        return sampled_idx.numpy()

    @torch.no_grad()
    def validation_epoch(self, model, san_check_step=-1, sample_events=True):
        """Validate one epoch.
        We aggregate the avg of all statistics and only log once.
        """
        super().validation_epoch(model, san_check_step=san_check_step)
        if self.local_rank != 0:
            return
        # visualization after every epoch
        if sample_events:
            self._sample_events(model)

    @staticmethod
    def event2video(events, caption=None, **quantize_args):
        """Convert events to wandb loggable videos."""
        imgs = events2frames(events, **quantize_args).astype(np.uint8)
        imgs = np.ascontiguousarray(imgs.transpose(0, 3, 1, 2))
        # add a black border to the video
        T, C, H, W = imgs.shape
        video = np.zeros((T, C, H + 8, W + 8), dtype=np.uint8)
        video[:, :, 4:-4, 4:-4] = imgs
        return wandb.Video(video, fps=2, caption=caption)

    def _configure_optimizers(self):
        """Returns an optimizer, a scheduler and its frequency (step/epoch)."""
        optimizer = super()._configure_optimizers()[0]

        if not self.params.lr_decay:
            return optimizer, (None, '')

        lr = self.params.lr
        total_steps = self.params.max_epochs * len(self.train_loader)
        warmup_steps = self.params.warmup_steps_pct * total_steps

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            total_steps,
            max_lr=lr,
            min_lr=lr / 100.,
            warmup_steps=warmup_steps,
        )

        return optimizer, (scheduler, 'step')


class EventCLIPMethod(EventBaseMethod):

    @torch.no_grad()
    def _sample_events(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        dst = self.val_loader.dataset
        dst.keep_events = True
        classes = dst.classes
        quantize_args = copy.deepcopy(dst.quantize_args)
        quantize_args['background_mask'] = True

        sampled_idx = self._get_sample_idx(self.params.n_samples, dst)
        log_dict = {}
        for i, idx in enumerate(sampled_idx):
            data_dict = dst[idx]
            events, label = data_dict.pop('events'), data_dict.pop('label')
            img, valid_mask = data_dict['img'], data_dict['valid_mask']
            # [N, 3, H, W], [N]
            in_dict = {
                'img': img[None].to(model.device),
                'valid_mask': valid_mask[None].to(model.device),
            }
            probs = model(in_dict)['probs'][0]  # [n_cls]

            # keep the topk predictions
            k = min(3, probs.shape[-1])
            topk = probs.topk(k, dim=-1)
            idxs, probs = \
                topk.indices.cpu().numpy(), topk.values.cpu().numpy()
            caption = f'GT: {classes[label]}\n' + '\t'.join([
                f'{classes[idx]}: {prob:.4f}'
                for idx, prob in zip(idxs, probs)
            ])

            # visualize the events
            # raw events
            raw_events = self.event2video(
                events, caption=caption, **quantize_args)
            log_dict[f'val/raw_events_{i}'] = raw_events
            # model inputs
            img = img[valid_mask]
            video = self._convert_video(img, caption=caption)
            log_dict[f'val/video_{i}'] = video

        wandb.log(log_dict, step=self.it)
        torch.cuda.empty_cache()
        dst.keep_events = False
