# Benchmark

## Overview

We provide instructions on reproducing the results reported in the paper, including:

-   Zero-shot EventCLIP on N-Caltech, N-Cars, and N-ImageNet datasets
-   Few-shot EventCLIP with text adapter (most stable), or joint adapter (best performing) on 3 datasets
-   Fine-tuning EventCLIP on N-Caltech and N-ImageNet to achieve SOTA performance
-   Learning with unlabeled data by self-training on generated pseudo labels on the N-ImageNet (Mini) dataset

In the following instructions, we will mostly use **EventCLIP with joint adapter on N-Caltech under the 5-shot setting** as example.
Other settings are easily replicable by changing the config file or other flags.

### Pre-trained Weights

Since most of the experiments in the paper can be trained within 1-2 hours, we only provide pre-trained weights for long-running experiments, or those involving multi-step training.
Please download the pre-trained weights from [Google Drive](https://drive.google.com/file/d/1QW7sn5BYjRdUe6xD_jQUQgSa9oIveq0s/view?usp=sharing) and unzip them under [pretrained/](../pretrained/).

## Training EventCLIP Feature Adapter

**We provide a unified script [train.py](../train.py) to train all models used in this project.**
You should always call it in the **root directory** of this repo (i.e. calling `python train.py xxx`).

**All of the model training can be done by specifying a config file (called `params` here), and adding other args (e.g. `--num_shots`).**
Please check the config file for the number of GPUs and other resources (e.g. `num_workers` CPUs) before launching a training.

Here is one example:

```
python train.py --params configs/fsclip/joint_adapter/joint_fsclip_ncaltech_params.py --num_shots 5 --fp16 --cudnn
```

Other arguments include:

-   `--weight`: resume training from this weight
-   `--ddp`: use DDP multi-GPU training (needed when using `>=2` GPUs)
-   `--fp16`: enable half-precision training (highly recommended)
-   `--cudnn`: enable cudnn benchmark (highly recommended)
-   `--local_rank`/`--local-rank`: required by DDP, don't change it

During training, model checkpoints and visualizations will be saved under `./checkpoint/$PARAMS/models/`.

We provide config files for training EventCLIP under the few-shot classification setting:

-   [Text Adapter configs](../configs/fsclip/text_adapter/)
-   [Joint Adapter configs](../configs/fsclip/joint_adapter/)

## Testing

Testing can be done with [test.py](../test.py).
To test the above trained model, simply run:

```
python test.py --params configs/fsclip/joint_adapter/joint_fsclip_ncaltech_params.py --weight $WEIGHT
```

Or you can use these configs to test the zero-shot classification performance:

-   [Zero-shot configs](../configs/zsclip/)

Other arguments in `test.py` include:

-   `--bs`: testing batch size
-   `--subset`: used to specify the N-ImageNet robustness variants to test. See their paper Appendix for a conversion between subset ID and the actual data variation
-   `--arch`: change CLIP's image encoder backbone in zero-shot testing
-   `--prompt`: change the text prompt in zero-shot testing

We also provide a `--train_shots` argument to automatically gather results over different shots.
If you train the same model with different `--num_shots` values in `train.py`, you can put all numbers of shots here to test them together.
For example, you can run:

```
python test.py --params configs/fsclip/joint_adapter/joint_fsclip_ncaltech_params.py --train_shots 20 10 5 3 1
```

**Note that testing is always conducted over the entire test set without few-shot filtering.**

## Fine-tuning EventCLIP Full Model

Fine-tuning EventCLIP is similar to training an adapter model.
But they require more GPU memory and training time.
We provide config files for fine-tuning EventCLIP under the few-shot and full data setting.
Please refer to them for detailed training requirement:

-   [Fine-tuning configs](../configs/ftclip/)

We provide the weight for our fine-tuned EventCLIP (with ViT-B/16 backbone) on the N-ImageNet dataset.

## Learning with Unlabeled Data

To generate pseudo labels on unlabeled data, please use [gen_data.py](../gen_data.py).
For example, you can use the zero-shot EventCLIP to generate pseudo labels on the N-ImageNet (Mini) dataset by:

```
python gen_data.py --params configs/zsclip/zsclip_nin_mini_params-vitb32.py --weight '' \
    --conf_thresh 0.999 --tta --tta_min_prob --tta_consistent --topk 30 \
    --save_path data/pseudo-N_Imagenet/vitb32_zs-tta-thresh_0999-top30 \
    --gt_shots -1
```

Here, we use a very high confidence threshold of `0.999` to filter predictions.
This is because the pre-trained CLIP model always makes over-confident predictions (likely due to the learned temperature parameter `\tau`).
Other arguments include:

-   `--tta`, `--tta_min_prob`, and `--tta_consistent` are the techniques introduced in the paper to further improve the label quality
-   `--topk 30` means we only select the top-30 most confident predictions for each class
-   `--save_path` indicates the path to save the generated dataset
-   `--gt_shots` specifies the number of labeled data used to train the model as `--weight`
    Since we are using the zero-shot model here, we set it to `-1` and `--weight` is empty

If you want to study the semi-supervised setting, where we have `X` labeled data and all the remaining unlabeled data, you can first pre-train an EventCLIP with joint adapter using the [provided config file](../configs/fsclip/joint_adapter/joint_fsclip_nin_mini_params-vitb32.py).
We provide the 1-, 3-, 5-, 10-, and 20-shot pre-trained weights in this setting.
Then, run `gen_data.py` again, but use the joint adapter's config file as `--params`, `--gt_shots X`, and `--weight` pointing to the pre-trained model's weight.
Also, please use a lower confidence threshold `--conf_thresh 0.5` as the few-shot EventCLIP is now calibrated.
An example command is:

```
python gen_data.py --params configs/fsclip/joint_adapter/joint_fsclip_nin_mini_params-vitb32.py \
    --weight pretrained/joint_fsclip_nin_mini_params-vitb32-1shot-pretrain.pth \
    --conf_thresh 0.5 --tta --tta_min_prob --tta_consistent --topk 30 \
    --save_path data/pseudo-N_Imagenet/vitb32_1shot-tta-thresh_05-top30 \
    --gt_shots 1
```

Finally, to train on this generated dataset (i.e. self-training), please modify the [config file](../configs/fsclip/joint_adapter/joint_fsclip_nin_mini_params-vitb32.py)'s `data_root` field to the `save_path` above and run `train.py`.
**Note that** you should set `--num_shots` to `X + topk`.
This is because we select the `topk` most confident predictions per class as pseudo labels, plus the `X` GT labels per class to train the model.

We provide the weight for our EventCLIP (with ViT-B/32 backbone) trained on zero-shot generated pseudo labels on the N-ImageNet (Mini) dataset.

## Useful Scripts

We provide helper scripts for Slurm cluster job submission, and train/test over multiple settings.

-   You can use [sbatch_run.sh](../scripts/sbatch_run.sh) to automatically generate a sbatch file and submit the job to slurm.
    Simply run:

```
GPUS=$NUM_GPU CPUS_PER_GPU=8 MEM_PER_CPU=5 QOS=$QOS \
    ./scripts/sbatch_run.sh $PARTITION $JOB_NAME \
    train.py none (if DDP then change `none` to `ddp`) --py_args...
```

Again using the same example, we can set `--py_args...` as (see the config file for the number of GPU/CPU to use)

```
--params configs/fsclip/joint_adapter/joint_fsclip_ncaltech_params.py \
    --num_shots 5 --fp16 --cudnn
```

Then this will be equivalent to running the above `python train.py xxx` command in CLI.

Note that `sbatch_run.sh` calls a `resubmit_failed_job.sh` script inside, which will monitor the job status and resubmit the job if it fails.

-   We provide a script to submit multiple runs of the same experiment with different random seeds to slurm.
    To use the script [dup_run_sbatch.sh](../scripts/dup_run_sbatch.sh), simply run:

```
GPUS=$NUM_GPU CPUS_PER_GPU=8 MEM_PER_CPU=5 QOS=$QOS REPEAT=$NUM_REPEAT \
    ./scripts/dup_run_sbatch.sh $PARTITION $JOB_NAME \
    train.py none $PARAMS --py_args...
```

The other parts are really the same as `sbatch_run.sh`.
The only difference is that we need to input the config file `$PARAMS` separately, so that the script will make several copies to it, and submit different jobs.

Again training the same model, duplicating `3` times, on `rtx6000` partition, and in the name of `joint_fsclip_ncaltech_params`, simply run:

```
GPUS=1 CPUS_PER_GPU=8 MEM_PER_CPU=5 QOS=normal REPEAT=3 \
    ./scripts/dup_run_sbatch.sh rtx6000 joint_fsclip_ncaltech_params \
    train.py none \
    configs/fsclip/joint_adapter/joint_fsclip_ncaltech_params.py \
    --num_shots 5 --fp16 --cudnn
```

The model weights will be saved under `./checkpoint/joint_fsclip_ncaltech_params-dup$X-5shot/`.

-   We provide scripts to train one EventCLIP under all five numbers of shots used in the paper [train_all_shots.sh](../scripts/train_all_shots.sh).
    For example, if you want to run the above `dup_run_sbatch.sh` command with `20, 10, 5, 3, 1` shots, simply wrap that command with this script by:

```
./scripts/train_all_shots.sh "GPUS=1 CPUS_PER_GPU=8 MEM_PER_CPU=5 QOS=normal REPEAT=3 ./scripts/dup_run_sbatch.sh rtx6000 joint_fsclip_ncaltech_params train.py none configs/fsclip/joint_adapter/joint_fsclip_ncaltech_params.py --fp16 --cudnn" 20 10 5 3 1
```

The model weights will be saved under `./checkpoint/joint_fsclip_ncaltech_params-dup$X-$Yshot/`.
See the `Testing` section above on how to efficiently test all these models with the `--train_shots` flag.

-   In zero-shot testing, we provide script to test EventCLIP with different ViT's image encoder architectures [test_all_arch.sh](../scripts/test_all_arch.sh).
    Simply wrap your testing command with this script, and add the arches you want to try

-   On N-ImageNet testing, we provide script to test EventCLIP over all robustness variants [test_all_subset.sh](../scripts/test_all_subset.sh).
    Simply wrap your testing command with this script, and add the subsets you want to try
