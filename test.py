"""EventCLIP testing script"""

import os
import sys
import importlib
import argparse

from tqdm import tqdm

import torch

import clip

from nerv.training import BaseDataModule
from nerv.utils import AverageMeter

from models import build_model
from datasets import build_dataset


@torch.no_grad()
def main(params):
    # have to load CLIP model first
    arch = params.clip_dict['arch']
    device = 'cuda'
    model, preprocess = clip.load(arch, device=device)

    # build dataset
    params.data_transforms = preprocess
    if args.subset > 0:
        test_set = build_dataset(params, val_only=True, subset=args.subset)
    else:
        test_set = build_dataset(params, val_only=True)
    is_nin = (params.dataset == 'n_imagenet')

    datamodule = BaseDataModule(
        params, train_set=None, val_set=test_set, use_ddp=False)
    test_loader = datamodule.val_loader

    # build model
    params.clip_dict['clip_model'] = model
    params.clip_dict['class_names'] = test_set.classes
    if not is_zs:
        params.adapter_dict['in_dim'] = model.visual.output_dim
    model = build_model(params)

    # load weight
    # don't load for zero-shot models
    if args.weight and not is_zs:
        model.load_weight(args.weight)
    model = model.cuda().eval()

    # test
    probs_acc_meter, logits_acc_meter = AverageMeter(), AverageMeter()
    if is_nin:
        probs_acc5_meter, logits_acc5_meter = AverageMeter(), AverageMeter()

    for data_dict in tqdm(test_loader):
        data_dict = {k: v.cuda() for k, v in data_dict.items()}
        out_dict = model(data_dict)
        labels = data_dict['label']

        # based on aggregated probs
        probs = out_dict['probs']
        probs_acc = (probs.argmax(dim=-1) == labels).float().mean().item()
        probs_acc_meter.update(probs_acc, labels.shape[0])

        # based on aggregated logits
        logits = out_dict['logits']
        logits_acc = (logits.argmax(dim=-1) == labels).float().mean().item()
        logits_acc_meter.update(logits_acc, labels.shape[0])

        # top5 accuracy
        if is_nin:
            probs_acc5 = (probs.topk(5, dim=-1).indices == labels[:, None]).\
                float().sum(dim=-1).mean().item()
            probs_acc5_meter.update(probs_acc5, labels.shape[0])
            logits_acc5 = (logits.topk(5, dim=-1).indices == labels[:, None]).\
                float().sum(dim=-1).mean().item()
            logits_acc5_meter.update(logits_acc5, labels.shape[0])

    print(f'\n\nTesting {args.params}')
    print(f'Model weight: {args.weight}')
    print(f'\tProbs-based accuracy@1: {probs_acc_meter.avg * 100.:.2f}%')
    print(f'\tLogits-based accuracy@1: {logits_acc_meter.avg * 100.:.2f}%\n')
    if not is_nin:
        return
    print(f'\tProbs-based accuracy@5: {probs_acc5_meter.avg * 100.:.2f}%')
    print(f'\tLogits-based accuracy@5: {logits_acc5_meter.avg * 100.:.2f}%\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EventCLIP')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--weight', type=str, default='', help='load weight')
    parser.add_argument('--arch', type=str, default='')
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--bs', type=int, default=-1)
    parser.add_argument('--subset', type=int, default=-1)
    args = parser.parse_args()

    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(os.path.dirname(args.params))
    params = importlib.import_module(os.path.basename(args.params))
    params = params.EventCLIPParams()

    # adjust params
    is_zs = (params.model == 'ZSCLIP')
    if args.arch:
        params.clip_dict['arch'] = args.arch
        assert is_zs, 'can only change ViT arch in zero-shot testing'
    if args.prompt:
        params.clip_dict['prompt'] = args.prompt
        assert is_zs, 'can only change text prompt in zero-shot testing'
    if args.bs > 0:
        params.val_batch_size = args.bs
    if args.subset > 0:
        assert params.dataset == 'n_imagenet', 'only N-ImageNet has subsets'

    main(params)
