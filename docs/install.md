# Install

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for environment setup:

```
conda create -n eventclip python=3.8.8
conda activate eventclip
```

Then install PyTorch which is compatible with your CUDA setting.
In our experiments, we use PyTorch 1.12.1 + CUDA 11.3 (the CUDA version is fine as long as it meets the requirement [here](https://pytorch.org/get-started/previous-versions/). PyTorch 2.0 is not tested but could also be compatible):

```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install pytorch-lightning==1.8.6
```

The codebase heavily relies on [nerv](https://github.com/Wuziyi616/nerv) for project template and Trainer.
You can easily install it by:

```
git clone git@github.com:Wuziyi616/nerv.git
cd nerv
git checkout v0.3.0  # tested with v0.3.0 release
pip install -e .
```

This will automatically install packages necessary for the project.
Additional packages are listed as follows:

```
pip install ftfy regex tqdm  # packages required by CLIP
pip install git+https://github.com/openai/CLIP.git  # install CLIP from OpenAI
```

Finally, clone this project by:

```
cd ..  # move out from nerv/
git clone git@github.com:Wuziyi616/EventCLIP.git
cd EventCLIP
```

We use [wandb](https://wandb.ai/) for logging, please run `wandb login` to log in.

## Possible Issues

-   In case you encounter any environmental issues, you can refer to the conda env file exported from my server [environment.yml](../environment.yml).
    You can install the same environment by `conda env create -f environment.yml`.
