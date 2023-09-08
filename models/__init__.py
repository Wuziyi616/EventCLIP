import copy

from .clip_cls import ZSCLIPClassifier, FSCLIPClassifier
from .clip_cls_ft import FTCLIPClassifier
from .semi_supervised import SemiSupervisedModel


def build_model(params):
    if params.model == 'ZSCLIP':
        return ZSCLIPClassifier(clip_dict=params.clip_dict, )
    elif params.model == 'FSCLIP':
        return FSCLIPClassifier(
            adapter_dict=params.adapter_dict,
            clip_dict=params.clip_dict,
            loss_dict=params.loss_dict,
        )
    elif params.model == 'FTCLIP':
        return FTCLIPClassifier(
            adapter_dict=params.adapter_dict,
            clip_dict=params.clip_dict,
            loss_dict=params.loss_dict,
        )
    elif params.model.startswith('SS-'):
        print('Building semi-supervised model...')
        params = copy.deepcopy(params)
        params.model = params.model[3:]
        teacher = build_model(params)
        student = build_model(params)
        return SemiSupervisedModel(
            student=student,
            teacher=teacher,
            ss_dict=params.ss_dict,
        )
    else:
        raise NotImplementedError(f'{params.model} is not implemented.')
