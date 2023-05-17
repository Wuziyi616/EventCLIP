# Benchmark

## Overview

We provide instructions on reproducing the results reported in the paper, including:

-   Zero-shot EventCLIP on N-Caltech, N-Cars and N-ImageNet
-   Few-shot EventCLIP with text adapter (most stable), or joint adapter (best performing) on 3 datasets

In the following instructions, we will use **EventCLIP with joint adapter on N-Caltech dataset under 5-shot setting** as the example.

## Training

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

Other arguments include:

-   `--bs`: testing batch size
-   `--subset`: used to specify the N-ImageNet robustness variants to test. See their paper Appendix for a conversion between subset ID and the actual data variation
-   `--arch`: change CLIP's image encoder backbone in zero-shot testing
-   `--prompt`: change the text prompt in zero-shot testing

**Note that testing is always conducted over the entire test set without few-shot filtering.**

## Scripts

We provide helper scripts for Slurm cluster job submission, and train/test over multiple settings.

-   You can use [sbatch_run.sh](../scripts/sbatch_run.sh) to automatically generate a sbatch file and submit the job to slurm.
    Simply running:

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

-   We provide a script to **submit multiple runs of the same experiment with different random seeds** to slurm.

To use the duplicate-run script [dup_run_sbatch.sh](../scripts/dup_run_sbatch.sh), simply do:

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

The model weights will be saved under `./checkpoint/joint_fsclip_ncaltech_params-dup$X/`.

-   We provide scripts to train one EventCLIP under all five numbers of shots used in the paper [train_all_shots.sh](../scripts/train_all_shots.sh).
    For example, if you want to run the above `dup_run_sbatch.sh` command with `20, 10, 5, 3, 1` shots, simply wrap that command with this script by:

```
./scripts/train_all_shots.sh "GPUS=1 CPUS_PER_GPU=8 MEM_PER_CPU=5 QOS=normal REPEAT=3 ./scripts/dup_run_sbatch.sh rtx6000 joint_fsclip_ncaltech_params train.py none configs/fsclip/joint_adapter/joint_fsclip_ncaltech_params.py --fp16 --cudnn" 20 10 5 3 1
```

The model weights will be saved under `./checkpoint/joint_fsclip_ncaltech_params-dup$X-$Yshot/`

-   In zero-shot testing, we provide script to test EventCLIP with different ViT's image encoder architectures [test_all_arch.sh](../scripts/test_all_arch.sh).
    Simply wrap your testing command with this script, and add the arches you want to try

-   On N-ImageNet testing, we provide script to test EventCLIP over all robustness variants [test_all_subset.sh](../scripts/test_all_subset.sh)
    Simply wrap your testing command with this script, and add the subsets you want to try
