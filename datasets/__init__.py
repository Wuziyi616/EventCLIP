import copy

from .caltech import build_n_caltech_dataset, NCaltech101
from .cars import build_n_cars_dataset, NCars
from .imagenet import build_n_imagenet_dataset, NImageNet
from .imagenet_mini import build_n_imagenet_mini_dataset, NImageNetMini
from .event2img import build_event2img_dataset, Event2ImageDataset
from .vis import events2frames


def build_dataset(params, val_only=False, **kwargs):
    tta = kwargs.pop('tta', False)
    dst = params.dataset
    ev_dst = eval(f'build_{dst}_dataset')(params, val_only=val_only, **kwargs)

    # adjust max-views for event2img conversion
    train_params = copy.deepcopy(params)
    val_params = copy.deepcopy(params)
    val_params.quantize_args['max_imgs'] = 10  # load all views for testing

    if val_only or kwargs.get('gen_data', False):
        return build_event2img_dataset(val_params, ev_dst, tta=tta)

    return build_event2img_dataset(
        train_params, ev_dst[0], augment=params.get('img_aug', False)), \
        build_event2img_dataset(val_params, ev_dst[1], augment=False)
