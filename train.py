"""EventCLIP training script"""

import os
import sys
import pwd
import importlib
import argparse
import wandb

import torch

import clip

from nerv.utils import mkdir_or_exist
from nerv.training import BaseDataModule

from models import build_model
from method import build_method
from datasets import build_dataset


def main(params):
    if args.fp16:
        print("INFO: using FP16 training!")
    if args.ddp:
        print("INFO: using DDP training!")

    # have to load CLIP model first
    arch = params.clip_dict.pop('arch')
    device = 'cuda'
    model, preprocess = clip.load(arch, device=device)

    # build dataset
    params.data_transforms = preprocess
    train_set, val_set = build_dataset(params)
    datamodule = BaseDataModule(
        params, train_set=train_set, val_set=val_set, use_ddp=params.ddp)

    # build model
    params.clip_dict['clip_model'] = model
    params.clip_dict['class_names'] = train_set.classes
    params.resolution = train_set.resolution
    params.class_names = train_set.classes
    if hasattr(params, 'adapter_dict'):
        params.adapter_dict['in_dim'] = model.visual.output_dim
    model = build_model(params)

    exp_name = os.path.basename(args.params)
    ckp_path = os.path.join(CHECKPOINT, exp_name, 'models')
    if args.local_rank == 0:
        mkdir_or_exist(os.path.dirname(ckp_path))

        # on clusters, quota is limited
        # soft link temp space for checkpointing
        if not os.path.exists(ckp_path):
            os.system(r'ln -s /checkpoint/{}/{}/ {}'.format(
                pwd.getpwuid(os.getuid())[0], SLURM_JOB_ID, ckp_path))

        wandb_name = f'{exp_name}-{SLURM_JOB_ID}'
        wandb.init(
            project=params.project,
            name=wandb_name,
            id=wandb_name,
            dir=ckp_path,
        )

    method = build_method(
        model=model,
        datamodule=datamodule,
        params=params,
        ckp_path=ckp_path,
        local_rank=args.local_rank,
        use_ddp=args.ddp,
        use_fp16=args.fp16,
    )

    method.fit(
        resume_from=args.weight, san_check_val_step=params.san_check_val_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EventCLIP')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--num_shots', type=int, default=-1)
    parser.add_argument('--weight', type=str, default='', help='load weight')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument('--cudnn', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()

    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(os.path.dirname(args.params))
    params = importlib.import_module(os.path.basename(args.params))
    params = params.EventCLIPParams()
    params.ddp = args.ddp

    if args.num_shots > 0:
        params.num_shots = args.num_shots
        args.params = args.params + f'-{args.num_shots}shot'

        # adjust the batch size since N-Cars only have 2 classes
        if params.dataset == 'n_cars':
            params.train_batch_size = min(
                params.num_shots * 2,  # 2 classes
                params.train_batch_size)

    SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')
    CHECKPOINT = './checkpoint/'
    if args.cudnn:
        torch.backends.cudnn.benchmark = True
    main(params)
