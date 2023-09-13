"""EventCLIP testing script"""

import os
import os.path as osp
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


def get_real_path(path):
    while osp.islink(path):
        path = os.readlink(path)
    return path


def find_key_from_value(d, v):
    for k, v_ in d.items():
        if v_ == v:
            return k
    return None


@torch.no_grad()
def main(params):
    # have to load CLIP model first
    arch = params.clip_dict['arch']
    device = 'cuda'
    model, preprocess = clip.load(arch, device=device)

    # build training dataset for generating pseudo labels
    params.data_transforms = preprocess
    tta = args.tta
    is_nin = ('n_imagenet' in params.dataset)
    if not is_nin:
        assert params.dataset == 'n_caltech'
    test_set = build_dataset(params, val_only=False, gen_data=True, tta=tta)

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
        print(f'Loading weight: {args.weight}')
    model = model.cuda().eval()

    # test
    all_acc_meter, sel_acc_meter = AverageMeter(), AverageMeter()
    class_names, labels = test_set.classes, test_set.event_dataset.labels
    total_num = len(labels)
    gt_class_cnt = {k: (labels == i).sum() for i, k in enumerate(class_names)}
    sel_class_cnt = {k: 0 for k in class_names}
    sel_correct_class_cnt = {k: 0 for k in class_names}
    pred_path2cls = {}

    conf_thresh = args.conf_thresh
    select_num = 0
    for data_dict in tqdm(test_loader):
        data_idx = data_dict.pop('data_idx').numpy()  # [B]
        data_dict = {k: v.cuda() for k, v in data_dict.items()}
        if tta:  # loaded data in shape [B, 4, N, ...]
            data_dict['img'] = data_dict['img'].flatten(0, 1)
            data_dict['valid_mask'] = data_dict['valid_mask'].flatten(0, 1)
        out_dict = model(data_dict)
        labels = data_dict['label']  # [B]

        # based on aggregated probs
        pred_probs = out_dict['probs']
        # TODO: aggregate probs in a better way
        if tta:
            probs = pred_probs.unflatten(0, (-1, 4))  # [B, 4, n_cls]
            tta_mask = torch.ones_like(labels).bool()  # [B]
            # predictions over 4 views should be consistent
            if args.tta_consistent:
                pred_cls = probs.argmax(dim=-1)  # [B, 4]
                tta_mask &= (pred_cls[:, 0] == pred_cls[:, 1]) & \
                    (pred_cls[:, 0] == pred_cls[:, 2]) & \
                    (pred_cls[:, 0] == pred_cls[:, 3])
            # the minimum confidence should be larger than conf_thresh
            if args.tta_min_prob:
                min_probs = probs.max(-1).values.min(-1).values
                tta_mask &= (min_probs > conf_thresh)
            probs = probs.mean(dim=1)  # [B, n_cls]
        else:
            probs = pred_probs
        probs_acc = (probs.argmax(dim=-1) == labels).float().mean().item()
        all_acc_meter.update(probs_acc, labels.shape[0])

        # only trust probs > conf_thresh
        max_probs, pred_labels = probs.max(dim=-1)
        sel_mask = (max_probs > conf_thresh)
        if tta:
            sel_mask &= tta_mask
        correct_mask = sel_mask & (pred_labels == labels)
        sel_acc = (pred_labels[sel_mask] == labels[sel_mask]).float().mean().item()
        sel_acc_meter.update(sel_acc, sel_mask.sum().item())
        select_num += sel_mask.sum().item()
        # update class cnt
        for i, lbl in enumerate(labels):
            if sel_mask[i].item():
                sel_class_cnt[class_names[lbl.item()]] += 1
                if correct_mask[i].item():
                    sel_correct_class_cnt[class_names[lbl.item()]] += 1
            if sel_mask[i].item():
                ev_path = test_set.labeled_files[data_idx[i]]
                pred_path2cls[ev_path] = class_names[pred_labels[i].item()]

    print('\nClass stats:')
    for k in class_names:
        gt_num, sel_num, correct_num = \
            gt_class_cnt[k], sel_class_cnt[k], sel_correct_class_cnt[k]
        print(f'\t{k}: GT {gt_num}, select {sel_num}, {correct_num} correct')

    print(f'\n\nTesting {args.params}')
    print(f'Model weight: {args.weight}')
    print(f'\tProbs-based accuracy@1: {all_acc_meter.avg * 100.:.2f}%')

    print(f'\nUsing confidence threshold: {conf_thresh}')
    print(f'\tSelected accuracy@1: {select_num}/{total_num} -- {sel_acc_meter.avg * 100.:.2f}%')

    if not save_path:
        return
    # save pseudo labels to a new dataset
    train_path = osp.join(save_path, 'extracted_train') if \
        is_nin else osp.join(save_path, 'training')
    assert not osp.exists(save_path), f'{save_path} already exists!'
    os.makedirs(train_path, exist_ok=True)
    for path, pred_cls in pred_path2cls.items():
        path = get_real_path(path)
        # path: xxx/N-Caltech101/training/airplanes/airplanes_150.npy
        #       xxx/N_Imagenet/extracted_train/n02114855/n02114855_15515.npz
        # pred_cls is a semantic class name
        # some class names might have been altered
        if test_set.event_dataset.new_cnames is not None:
            ori_cls = find_key_from_value(
                test_set.event_dataset.new_cnames, pred_cls)
            if ori_cls is not None:
                pred_cls = ori_cls
        if is_nin:
            folder_name = test_set.event_dataset.name2folder[pred_cls]
        else:
            folder_name = pred_cls
        ev_name = osp.basename(path)
        # save to train_path/folder_name/ev_name
        # use soft link to save disk space
        new_path = osp.join(train_path, folder_name, ev_name)
        os.makedirs(osp.dirname(new_path), exist_ok=True)
        os.symlink(path, new_path)
    # also soft-link val/test set if they exist
    if is_nin:
        val_path = osp.join(save_path, 'extracted_val')
        ori_val_path = osp.join(osp.dirname(test_set.root), 'extracted_val')
        ori_val_path = get_real_path(ori_val_path)
        os.symlink(ori_val_path, val_path)
    else:
        val_path = osp.join(save_path, 'validation')
        test_path = osp.join(save_path, 'testing')
        ori_val_path = osp.join(osp.dirname(test_set.root), 'validation')
        ori_test_path = osp.join(osp.dirname(test_set.root), 'testing')
        ori_val_path = get_real_path(ori_val_path)
        ori_test_path = get_real_path(ori_test_path)
        os.symlink(ori_val_path, val_path)
        os.symlink(ori_test_path, test_path)
    print(f'\nSaved pseudo labels to {save_path}')
    # some classes don't have any pseudo labels
    # we still create the folder for consistency
    for k in class_names:
        if test_set.event_dataset.new_cnames is not None:
            ori_cls = find_key_from_value(test_set.event_dataset.new_cnames, k)
            if ori_cls is not None:
                k = ori_cls
        if is_nin:
            folder_name = test_set.event_dataset.name2folder[k]
        else:
            folder_name = k
        folder_path = osp.join(train_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EventCLIP')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--weight', type=str, default='', help='load weight')
    parser.add_argument('--conf_thresh', type=float, default=-1.)
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--tta_consistent', action='store_true')
    parser.add_argument('--tta_min_prob', action='store_true')
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--bs', type=int, default=-1)
    args = parser.parse_args()

    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(osp.dirname(args.params))
    params = importlib.import_module(osp.basename(args.params))
    params = params.EventCLIPParams()

    # adjust params
    is_zs = (params.model == 'ZSCLIP')
    if args.prompt:
        params.clip_dict['prompt'] = args.prompt
        assert is_zs, 'can only change text prompt in zero-shot testing'
    if args.bs > 0:
        params.val_batch_size = args.bs
    save_path = args.save_path
    if save_path:
        assert not osp.exists(save_path), f'{save_path} already exists!'

    main(params)
    exit(-1)
