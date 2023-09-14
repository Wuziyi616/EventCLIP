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


def print_stats(class_names, gt_class_cnt, sel_class_cnt,
                sel_correct_class_cnt):
    print('\nClass stats:')
    for k in class_names:
        gt_num, sel_num, correct_num = \
            gt_class_cnt[k], sel_class_cnt[k], sel_correct_class_cnt[k]
        print(f'\t{k}: GT {gt_num}, select {sel_num}, {correct_num} correct')
    print('Not accurate classes')
    less_accurate_cnt = 0
    for k in class_names:
        gt_num, sel_num, correct_num = \
            gt_class_cnt[k], sel_class_cnt[k], sel_correct_class_cnt[k]
        ratio = correct_num / sel_num if sel_num > 0 else 0.
        if ratio < 0.5:
            print(f'\t{k}: GT {gt_num}, select {correct_num}/{sel_num} -- {ratio:.2f}')
            less_accurate_cnt += 1
    print(f'Not accurate classes: {less_accurate_cnt}/{len(class_names)}')

    total_num = sum(gt_class_cnt.values())
    select_num = sum(sel_class_cnt.values())
    select_correct_num = sum(sel_correct_class_cnt.values())
    sel_acc = select_correct_num / select_num * 100. if select_num > 0 else 0.
    print(f'\nUsing {args.conf_thresh=}')
    if args.num_shots > 0:
        print(f'Using {args.num_shots=}')
    print(f'\tSelect {select_num} from {total_num}, Acc={sel_acc:.2f}%')
    if args.tta:
        print(f'Using TTA with {args.tta_consistent=} + {args.tta_min_prob=}')


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
        assert params.dataset == 'n_caltech', f'{params.dataset} not supported'
    print(f'Generate pseudo labels for {params.dataset}')
    test_set = build_dataset(params, val_only=False, gen_data=True, tta=tta)
    ev_dst = test_set.event_dataset

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
    all_acc_meter = AverageMeter()
    class_names, labels = test_set.classes, ev_dst.labels
    # total_num = len(labels)
    gt_class_cnt = {k: (labels == i).sum() for i, k in enumerate(class_names)}
    sel_class_cnt = {k: 0 for k in class_names}
    sel_correct_class_cnt = {k: 0 for k in class_names}
    pred_path2cls = {}

    conf_thresh = args.conf_thresh
    num_shots = args.num_shots
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
        # update class cnt
        for i, (lbl, pred_lbl) in enumerate(zip(labels, pred_labels)):
            pred_cls_name = class_names[pred_lbl.item()]
            if sel_mask[i].item():
                sel_class_cnt[pred_cls_name] += 1
                if pred_lbl.item() == lbl.item():
                    sel_correct_class_cnt[pred_cls_name] += 1
            if sel_mask[i].item():
                ev_path = str(ev_dst.labeled_files[data_idx[i]])
                if num_shots > 0:  # also record the probs, take top-k later
                    pred_path2cls[ev_path] = {
                        'cls': pred_cls_name,
                        'prob': max_probs[i].item(),
                    }
                else:
                    pred_path2cls[ev_path] = pred_cls_name

    print_stats(class_names, gt_class_cnt, sel_class_cnt,
                sel_correct_class_cnt)
    print(f'\n\nTesting {args.params}')
    if args.weight:
        print(f'Model weight: {args.weight}')
    print(f'\tProbs-based accuracy@1: {all_acc_meter.avg * 100.:.2f}%')

    if not save_path:
        return
    # save pseudo labels to a new dataset
    train_path = osp.join(save_path, 'extracted_train') if \
        is_nin else osp.join(save_path, 'training')
    assert not osp.exists(save_path), f'{save_path} already exists!'
    os.makedirs(train_path, exist_ok=True)
    # some classes might be renamed
    new_cnames = ev_dst.new_cnames
    # only take top-`num_shots` predictions for each class
    if num_shots > 0:
        topk_pred_path2cls, sel_class_cnt, sel_correct_class_cnt = {}, {}, {}
        for cls_name in class_names:
            sel_correct_class_cnt[cls_name] = 0
            cls_pred_paths, cls_pred_probs = [], []
            # find data that are classified as this class
            for path, pred in pred_path2cls.items():
                if pred['cls'] == cls_name:
                    cls_pred_paths.append(path)
                    cls_pred_probs.append(pred['prob'])
            cls_pred_probs = torch.tensor(cls_pred_probs)
            topk = min(num_shots, cls_pred_probs.shape[0])
            _, topk_idx = cls_pred_probs.topk(topk)
            for i in topk_idx:
                path = cls_pred_paths[i]  # get the GT label from path
                gt_cls_name = osp.basename(osp.dirname(path))
                if is_nin:
                    gt_cls_name = ev_dst.folder2name[gt_cls_name]
                if new_cnames is not None:
                    gt_cls_name = new_cnames.get(gt_cls_name, gt_cls_name)
                if gt_cls_name == cls_name:
                    sel_correct_class_cnt[cls_name] += 1
                topk_pred_path2cls[path] = cls_name
            sel_class_cnt[cls_name] = topk
        pred_path2cls = topk_pred_path2cls
        print_stats(class_names, gt_class_cnt, sel_class_cnt,
                    sel_correct_class_cnt)
    # save pseudo labels
    for path, pred_cls in pred_path2cls.items():
        path = get_real_path(path)
        # path: xxx/N-Caltech101/training/airplanes/airplanes_150.npy
        #       xxx/N_Imagenet/extracted_train/n02114855/n02114855_15515.npz
        # pred_cls is a semantic class name
        # some class names might have been altered
        if new_cnames is not None:
            ori_cls = find_key_from_value(new_cnames, pred_cls)
            if ori_cls is not None:
                pred_cls = ori_cls
        if is_nin:
            folder_name = ev_dst.name2folder[pred_cls]
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
        ori_val_path = osp.join(osp.dirname(ev_dst.root), 'extracted_val')
        ori_val_path = get_real_path(ori_val_path)
        os.symlink(ori_val_path, val_path)
    else:
        val_path = osp.join(save_path, 'validation')
        test_path = osp.join(save_path, 'testing')
        ori_val_path = osp.join(osp.dirname(ev_dst.root), 'validation')
        ori_test_path = osp.join(osp.dirname(ev_dst.root), 'testing')
        ori_val_path = get_real_path(ori_val_path)
        ori_test_path = get_real_path(ori_test_path)
        os.symlink(ori_val_path, val_path)
        os.symlink(ori_test_path, test_path)
    print(f'\nSaved pseudo labels to {save_path}')
    # some classes don't have any pseudo labels
    # we still create the folder for consistency
    for k in class_names:
        if new_cnames is not None:
            ori_cls = find_key_from_value(new_cnames, k)
            if ori_cls is not None:
                k = ori_cls
        if is_nin:
            folder_name = ev_dst.name2folder[k]
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
    parser.add_argument('--num_shots', type=int, default=-1)
    args = parser.parse_args()

    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(osp.dirname(args.params))
    params = importlib.import_module(osp.basename(args.params))
    params = params.EventCLIPParams()

    # adjust params
    is_zs = (params.model == 'ZSCLIP')
    save_path = args.save_path
    if save_path:
        assert not osp.exists(save_path), f'{save_path} already exists!'

    main(params)
    exit(-1)
