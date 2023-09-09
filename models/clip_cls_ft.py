import copy

import torch
import torch.nn as nn
from torch.nn import functional as F

import clip

from nerv.training import BaseModel

from .adapter import IdentityAdapter, TransformerAdapter
from .lora import inject_trainable_lora


class FTCLIPClassifier(BaseModel):
    """Finetune CLIP model for **few-shot** classification."""

    def __init__(
            self,
            adapter_dict=dict(
                adapter_type='text-identity',
                residual=True,
            ),
            clip_dict=dict(
                clip_model=None,
                prompt='a point cloud image of a {}',
                class_names=None,
                agg_func='sum',
            ),
            loss_dict=dict(
                use_logits_loss=True,
                use_probs_loss=False,
            ),
    ):
        super().__init__()

        self.clip_dict = clip_dict
        self.loss_dict = loss_dict
        self.adapter_dict = copy.deepcopy(adapter_dict)

        self._build_clip()
        self._build_loss()
        self._build_adapter()

    def _build_clip(self):
        # freeze the CLIP model
        model = self.clip_dict['clip_model']
        for p in model.parameters():
            p.requires_grad = False
        # LoRA fine-tuning
        lora = self.clip_dict.get('lora', -1)
        if isinstance(lora, str) or lora > 0:
            model.visual = inject_trainable_lora(model.visual, r=lora)
        # finetune CLIP.visual or its sub-layers
        conv1 = self.clip_dict['only_conv1']
        bias = self.clip_dict['only_bias']
        ln = self.clip_dict['only_ln']
        cls_fc = self.clip_dict.get('only_cls_fc', False)
        cls_token = self.clip_dict.get('only_cls_token', False)
        if conv1:  # only tune the first conv layer
            for p in model.visual.conv1.parameters():
                p.requires_grad = True
        if bias:  # only tune the bias terms
            for name, p in model.visual.named_parameters():
                if 'bias' in name and p is not None:
                    p.requires_grad = True
        if ln:  # only tune the LayerNorm layers
            for m in model.visual.modules():
                if isinstance(m, nn.LayerNorm):
                    for p in m.parameters():
                        p.requires_grad = True
        if cls_fc:  # only tune the final projection head
            model.visual.proj.requires_grad = True
        if cls_token:  # only tune the CLS token
            model.visual.class_embedding.requires_grad = True
        # tune all
        if (isinstance(lora, int) and lora <= 0) and \
                not (conv1 or bias or ln or cls_fc or cls_token):
            for p in model.visual.parameters():
                p.requires_grad = True
        # set as eval
        self.model = model.eval()
        self.logit_scale = model.logit_scale.exp().item()

        # text prompt for zero-shot cls
        self.prompt = self.clip_dict['prompt']
        self.class_names = self.clip_dict['class_names']
        self.text_feats = None

        # aggregation function
        self.agg_func = self.clip_dict['agg_func']
        assert self.agg_func in ['sum', 'mean', 'max']

    def _build_loss(self):
        self.use_logits_loss = self.loss_dict['use_logits_loss']
        self.use_probs_loss = self.loss_dict['use_probs_loss']
        assert int(self.use_logits_loss) + int(self.use_probs_loss) == 1

    def _build_prompts(self, adapter_type):
        """Build the text features for prompt tuning."""
        with torch.no_grad():
            text_feats = self._get_text_feats().float()  # [n_classes, C]
        self.text_feats = nn.Parameter(text_feats, requires_grad=True)
        adapter_type = adapter_type[5:]
        return adapter_type

    def _build_adapter(self):
        # whether to tune the text features as well
        adapter_type = self.adapter_dict.pop('adapter_type').lower()
        if adapter_type.startswith('text-'):
            print('Tune text features as well!')
            self.prompt_tuning = True
            adapter_type = self._build_prompts(adapter_type)
        else:
            self.prompt_tuning = False

        # image feature adapter
        self.adapter_type = adapter_type
        assert adapter_type == 'identity'
        if adapter_type == 'identity':  # not tuning image features
            model = IdentityAdapter
        elif adapter_type == 'trans':  # Transformer to fuse image features
            model = TransformerAdapter
        else:
            raise NotImplementedError(f'adapter {adapter_type} not supported!')
        self.adapter = model(**self.adapter_dict)

    def _same_class_names(self, class_names):
        """Check if the input `class_names` matches `self.class_names`."""
        return all([c1 == c2 for c1, c2 in zip(class_names, self.class_names)])

    def _get_text_feats(self, class_names=None):
        """Compute the text prompt features using CLIP text encoder."""
        # no `class_names` provided
        if class_names is None:
            no_cls_flag = True
            class_names = self.class_names
            # with cached `text_feats`
            if self.text_feats is not None:
                return self.text_feats
        # `class_names` matches
        elif self._same_class_names(class_names):
            # with cached `text_feats`
            if self.text_feats is not None:
                return self.text_feats

        # compute the text prompt features
        class_names = [c.lower().replace('_', ' ') for c in class_names]
        prompts = torch.cat([
            clip.tokenize(self.prompt.format(c)) for c in class_names
        ]).to(self.device)
        text_feats = self.model.encode_text(prompts)
        text_feats = F.normalize(text_feats, p=2, dim=-1)

        # cache the `text_feats` if
        # 1) the `class_names` matches
        # 2) the `class_names` is not provided
        if no_cls_flag or self._same_class_names(class_names):
            self.text_feats = text_feats

        return text_feats  # [n_classes, C]

    def get_text_feats(self, class_names=None):
        # finetune the text features (i.e. prompt tuning)
        if self.prompt_tuning:
            assert self.text_feats.requires_grad, 'prompt should be trainable!'
            text_feats = F.normalize(self.text_feats, p=2, dim=-1)
        # otherwise, we use fixed text features
        else:
            with torch.no_grad():
                text_feats = self._get_text_feats(class_names)
        return self._adjust_dtype(text_feats)

    def _get_img_feats(self, imgs):
        """Compute the image features using CLIP image encoder.

        Args:
            imgs (torch.Tensor): [B, C, H, W]
        """
        img_feats = self.model.encode_image(imgs)
        return img_feats  # [B, C]

    def get_img_feats(self, imgs):
        img_feats = self._get_img_feats(imgs)
        return self._adjust_dtype(img_feats)

    def _aggregate_logits(self, logits, valid_masks):
        """Aggregate logits for each data.

        Args:
            logits (torch.Tensor): [B, T, n_classes]
            valid_masks (torch.Tensor): [B, T]
        """
        if self.agg_func == 'sum':
            logits = logits.sum(1)
        elif self.agg_func == 'mean':
            logits = logits.sum(1) / valid_masks.float().sum(1, keepdim=True)
        elif self.agg_func == 'max':
            # make invalid logits very small
            logits = logits - (1. - valid_masks.float()) * 1e6
            logits = logits.max(1)[0]
        else:
            raise NotImplementedError
        return logits

    def _aggregate_probs(self, logits, valid_masks):
        """This one always take the mean."""
        valid_masks = valid_masks.detach().float()
        probs = logits.softmax(dim=-1)
        probs = probs * valid_masks[..., None]
        probs = probs.sum(1) / valid_masks.sum(1, keepdim=True)
        return probs

    def forward(self, data_dict):
        """Forward function."""
        imgs = data_dict['img']  # [B, T, C, H, W], `T` is number of views
        valid_masks = data_dict['valid_mask']  # [B, T]
        B, T = valid_masks.shape

        # compute image features
        valid_imgs = imgs[valid_masks]  # [N, C, H, W]
        img_feats = self.get_img_feats(valid_imgs)  # [N, C]

        # update image features using adapter
        C = img_feats.shape[-1]
        full_img_feats = torch.zeros(B, T, C).type_as(img_feats)
        full_img_feats[valid_masks] = img_feats
        # full_img_feats = self.adapter(full_img_feats, valid_masks)
        # [B, T, C], multi-view image features
        # normalize the output features
        # all zeros vector will still be zeros after F.normalize()
        full_img_feats = F.normalize(
            full_img_feats, p=2, dim=-1).type_as(full_img_feats)
        # make invalid features zeros
        full_img_feats = full_img_feats * valid_masks.float().unsqueeze(-1)

        # compute text features
        # we may need to compute gradients w.r.t. text features
        # so we can't use torch.no_grad() here
        text_feats = self.get_text_feats()  # [n_classes, C]

        # compute logits
        full_logits = (self.logit_scale * full_img_feats @ text_feats.T)
        # [B, T, n_cls], multi-view logits

        # convert to [B, n_cls] for loss computation!
        logits = self._aggregate_logits(full_logits, valid_masks)
        probs = self._aggregate_probs(full_logits, valid_masks)

        out_dict = {
            'full_logits': full_logits,  # [B, T, n_classes]
            'valid_masks': valid_masks,  # [B, T]
            'logits': logits,  # [B, n_classes]
            'probs': probs,  # [B, n_classes]
        }
        return out_dict

    def calc_train_loss(self, data_dict, out_dict):
        """Compute training loss."""
        labels = data_dict['label']  # [B]
        logits = out_dict['logits']  # [B, n_classes]
        probs = out_dict['probs']  # [B, n_classes]
        loss_dict = {}
        if self.use_logits_loss:
            loss_dict['ce_loss'] = F.cross_entropy(logits, labels)
        if self.use_probs_loss:
            probs = probs + 1e-6  # avoid nan
            loss_dict['ce_loss'] = F.nll_loss(probs.log(), labels)
        return loss_dict

    @torch.no_grad()
    def calc_eval_loss(self, data_dict, out_dict):
        """Loss computation in eval."""
        loss_dict = self.calc_train_loss(data_dict, out_dict)

        # also compute the cls accuracy
        labels = data_dict['label']  # [B]
        # based on aggregated probs
        probs = out_dict['probs']  # [B, n_classes]
        probs_acc = (probs.argmax(dim=-1) == labels).float().mean()
        loss_dict['probs_acc'] = probs_acc
        # based on aggregated logits
        logits = out_dict['logits']  # [B, n_classes]
        logits_acc = (logits.argmax(dim=-1) == labels).float().mean()
        loss_dict['logits_acc'] = logits_acc
        return loss_dict

    def _adjust_dtype(self, x):
        """CLIP model returns features in FP16.
        During training, torch.amp will help us handle this.
        However, during inference, we need to manually convert them to FP32.
        """
        if self.training:
            return x
        return x.type(self.dtype)

    @property
    def dtype(self):
        return self.adapter.dtype

    @property
    def device(self):
        return self.model.logit_scale.device

    def train(self, mode=True):
        nn.Module.train(self, mode)
        # keep CLIP in eval mode
        self.model.eval()
        # but adjust CLIP.visual
        self.model.visual.train(mode)
        return self

    def state_dict(self):
        """Remove CLIP weight (keep `model.visual`) from the state dict."""
        w = super().state_dict()
        w = {
            k: v
            for k, v in w.items()
            if ((not k.startswith('model.')) or k.startswith('model.visual.'))
        }
        return w

    def load_state_dict(self, state_dict, strict=True):
        """Don't load CLIP weight (load `model.visual`) from the state dict."""
        # load CLIP weight from the state dict
        clip_w = {
            f'model.{k}': v
            for k, v in self.model.state_dict().items()
            if not k.startswith('visual.')
        }
        assert all(k not in state_dict for k in clip_w)
        state_dict = {**clip_w, **state_dict}
        super().load_state_dict(state_dict, strict=strict)
