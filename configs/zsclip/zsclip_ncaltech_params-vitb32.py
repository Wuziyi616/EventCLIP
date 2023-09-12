from nerv.training import BaseParams


class EventCLIPParams(BaseParams):
    project = 'EventCLIP'

    # training settings
    gpus = 1

    # data settings
    dataset = 'n_caltech'
    data_root = './data/N-Caltech101/'
    train_batch_size = 32 // gpus
    val_batch_size = train_batch_size * 2
    num_workers = 8

    # event2img conversion
    quantize_args = dict(
        max_imgs=2,
        N=20000,
        split_method='event_count',
        convert_method='event_histogram',
        grayscale=True,
        count_non_zero=False,
        background_mask=True,
    )

    # model configs
    model = 'ZSCLIP'
    clip_dict = dict(
        # 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32'
        # 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
        arch='ViT-B/32',
        prompt='a sketch image of a {}',
        agg_func='mean',  # aggregate the logits over views
    )
