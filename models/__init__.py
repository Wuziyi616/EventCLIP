from .clip_cls import ZSCLIPClassifier, FSCLIPClassifier
from .clip_cls_ft import FTCLIPClassifier


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
    else:
        raise NotImplementedError(f'{params.model} is not implemented.')
