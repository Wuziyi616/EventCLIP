import torch
import torch.nn as nn
import torch.nn.functional as F

from nerv.training import BaseModel


@torch.no_grad()
def ema_model_update(model, ema_model, global_step, alpha=0.999):
    assert 0. < alpha < 1.
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        if param.requires_grad:
            ema_param.data.mul_(alpha).add_(param.data, alpha=1. - alpha)


@torch.no_grad()
def copy_model_params(src_model, model):
    """Copy the parameters from src_model to tgt_model."""
    for src_param, param in zip(src_model.parameters(), model.parameters()):
        if src_param.requires_grad:
            param.data.copy_(src_param.data)


class SemiSupervisedModel(BaseModel):
    """Student-Teacher semi-supervised model."""

    def __init__(
        self,
        student,
        teacher,
        ss_dict=dict(
            topk=16,  # take top-K preds
            conf_thresh=0.0,  # take preds with conf > thresh
            use_ema=True,
            ema_alpha=0.999,
        ),
    ):
        super().__init__()

        self.student = student
        teacher.load_state_dict(student.state_dict())
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.topk = ss_dict['topk']
        assert isinstance(self.topk, int) and self.topk > 0
        self.conf_thresh = ss_dict['conf_thresh']
        assert 0. <= self.conf_thresh <= 1.
        self.use_ema = ss_dict['use_ema']
        self.ema_alpha = ss_dict['ema_alpha']

    @torch.no_grad()
    def _select_labels(self, max_probs):
        """Select the pseudo labels with high confidence."""
        # topk OR conf > thresh
        conf_mask = (max_probs > self.conf_thresh)
        if conf_mask.sum() >= self.topk:
            return conf_mask
        conf_mask = torch.zeros_like(max_probs).bool()
        idxs = max_probs.topk(self.topk, dim=0, largest=True, sorted=False)[1]
        conf_mask[idxs] = True
        return conf_mask

    @torch.no_grad()
    def _generate_labels(self, imgs, valid_masks, real_labels):
        """Generate pseudo labels on unlabeled samples."""
        assert (real_labels >= 0).all()
        unsup_data_dict = {
            'img': imgs,
            'valid_mask': valid_masks,
        }
        pred_probs = self.teacher(unsup_data_dict)['probs']  # [n, n_cls]
        max_probs, pred_labels = pred_probs.max(dim=-1)  # [n]
        all_acc = (pred_labels == real_labels).float().mean()
        # only train on confident pseudo labels
        conf_mask = self._select_labels(max_probs)  # [n]
        if conf_mask.any():
            select_acc = (pred_labels[conf_mask] == real_labels[conf_mask]
                          ).float().mean()
        else:
            select_acc = torch.tensor(0.).type_as(all_acc)
        log_dict = {
            'unlabeled_acc': all_acc,
            'unlabeled_num': torch.tensor(len(real_labels)).type_as(all_acc),
            'select_acc': select_acc,
            'select_num': conf_mask.float().sum(),
            'min_select_probs': max_probs[conf_mask].min(),
        }
        # also compute acc of student model (ideally should be < teacher)
        self.student.eval()
        student_probs = self.student(unsup_data_dict)['probs']  # [n, n_cls]
        student_acc = (student_probs.argmax(-1) == real_labels).float().mean()
        log_dict['student_unlabeled_acc'] = student_acc
        self.student.train()
        return pred_labels, conf_mask, log_dict

    def forward(self, data_dict):
        """Forward function."""
        if not self.training:
            return self.teacher(data_dict)

        weak_imgs = data_dict['weak_img']  # [B, T, C, H, W]
        strong_imgs = data_dict['strong_img']  # [B, T, C, H, W]
        valid_masks = data_dict['valid_mask']  # [B, T]
        labels = data_dict['label']  # [B]

        # generate pseudo labels on unlabeled samples
        # loaded unsup_label = -1 * real_label - 1
        # so real_label = -1 * unsup_label - 1
        unsup_mask = (labels < 0)  # [B]
        sup_mask = (~unsup_mask)  # [B]
        if unsup_mask.any():
            real_labels = -1 * labels[unsup_mask] - 1  # [n]
            pred_labels, conf_mask, log_dict = self._generate_labels(
                weak_imgs[unsup_mask], valid_masks[unsup_mask], real_labels)
            unsup_mask[unsup_mask.clone()] = conf_mask
            # update label with high-conf pseudo labels
            labels[unsup_mask] = pred_labels[conf_mask]
        # gather data to train on
        train_mask = (sup_mask | unsup_mask)  # [B]
        train_data_dict = {
            'img': strong_imgs[train_mask],
            'valid_mask': valid_masks[train_mask],
        }
        pse_labels = labels[train_mask]
        assert (pse_labels >= 0).all()

        # train the student model
        out_dict = self.student(train_data_dict)
        # add pseudo labels and accuracy to out_dict
        out_dict['pse_labels'] = pse_labels
        out_dict['log_dict'] = log_dict
        return out_dict

    def calc_train_loss(self, data_dict, out_dict):
        """Compute training loss."""
        labels = out_dict['pse_labels']  # [B']
        logits = out_dict['logits']  # [B', n_classes]
        probs = out_dict['probs']  # [B', n_classes]
        loss_dict = out_dict.pop('log_dict')
        with torch.no_grad():
            loss_dict['acc'] = (probs.argmax(-1) == labels).float().mean()
        if self.student.use_logits_loss:
            loss_dict['ce_loss'] = F.cross_entropy(logits, labels)
        if self.student.use_probs_loss:
            probs = probs + 1e-6  # avoid nan
            loss_dict['ce_loss'] = F.nll_loss(probs.log(), labels)
        return loss_dict

    @torch.no_grad()
    def calc_eval_loss(self, data_dict, out_dict):
        """Loss computation in eval."""
        return self.teacher.calc_eval_loss(data_dict, out_dict)

    def _training_step_end(self, method=None):
        """Things to do at the end of every training step."""
        if self.use_ema:
            global_step = method.it
            ema_model_update(
                self.student, self.teacher, global_step, alpha=self.ema_alpha)
        else:
            copy_model_params(self.student, self.teacher)

    def train(self, mode=True):
        nn.Module.train(self, mode)
        # keep the teacher model in eval mode
        self.teacher.eval()
        return self

    @property
    def dtype(self):
        return self.teacher.dtype

    @property
    def device(self):
        return self.teacher.device

    def state_dict(self):
        student_w = self.student.state_dict()
        teacher_w = self.teacher.state_dict()
        w = {f'student.{k}': v for k, v in student_w.items()}
        w.update({f'teacher.{k}': v for k, v in teacher_w.items()})
        return w

    def load_state_dict(self, state_dict, strict=True):
        student_w = {
            k[8:]: v
            for k, v in state_dict.items() if k.startswith('student.')
        }
        teacher_w = {
            k[8:]: v
            for k, v in state_dict.items() if k.startswith('teacher.')
        }
        self.student.load_state_dict(student_w, strict=strict)
        self.teacher.load_state_dict(teacher_w, strict=strict)
