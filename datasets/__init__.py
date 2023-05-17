import copy

from .caltech import build_n_caltech_dataset, NCaltech101
from .cars import build_n_cars_dataset, NCars
from .imagenet import build_n_imagenet_dataset, NImageNet
from .event2img import build_event2img_dataset, Event2ImageDataset
from .vis import events2frames


def build_dataset(params, val_only=False, **kwargs):
    dst = params.dataset
    event_dataset = eval(f'build_{dst}_dataset')(
        params, val_only=val_only, **kwargs)

    # adjust max-views for event2img conversion
    train_params = copy.deepcopy(params)
    val_params = copy.deepcopy(params)
    val_params.quantize_args['max_imgs'] = 10  # load all views for testing

    if val_only:
        return build_event2img_dataset(val_params, event_dataset)

    return build_event2img_dataset(train_params, event_dataset[0]), \
        build_event2img_dataset(val_params, event_dataset[1])
