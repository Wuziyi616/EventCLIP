from nerv.training import BaseParams


class EventCLIPParams(BaseParams):
    project = 'EventCLIP'

    # training settings
    gpus = 1
    max_epochs = 100
    save_interval = 1
    eval_interval = 5
    save_epoch_end = True
    n_samples = 10

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 2e-5
    warmup_steps_pct = 0.05
    lr_decay = 'cosine'

    # data settings
    dataset = 'n_imagenet'
    data_root = './data/N_Imagenet/'
    num_shots = None
    train_batch_size = 128 // gpus
    val_batch_size = train_batch_size * 2
    num_workers = 16

    # event2img conversion
    quantize_args = dict(
        max_imgs=2,
        N=70000,
        split_method='event_count',
        convert_method='event_histogram',
        grayscale=True,
        count_non_zero=False,
        background_mask=True,
    )

    # model configs
    model = 'FSCLIP'
    clip_dict = dict(
        # 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32'
        # 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
        arch='ViT-L/14',
        prompt='a point cloud image of a {}',
        agg_func='mean',  # aggregate the logits over views
    )

    # adapter configs
    d_model = 256
    adapter_dict = dict(
        adapter_type='text-trans',
        in_dim=512,
        d_model=d_model,
        num_heads=d_model // 64,
        ffn_dim=d_model * 4,
        norm_first=True,
        num_layers=2,
        residual=0.95,
    )

    # loss configs
    loss_dict = dict(
        use_logits_loss=True,  # CE over mean logits
        use_probs_loss=False,  # CE over mean probs
    )

    ce_loss_w = 1.
