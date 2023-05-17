# EventCLIP

[**EventCLIP: Adapting CLIP for Event-based Object Recognition**](https://github.com/Wuziyi616/EventCLIP)<br/>
[Ziyi Wu](https://wuziyi616.github.io/),
Xudong Liu,
[Igor Gilitschenski](https://tisl.cs.utoronto.ca/author/igor-gilitschenski/)<br/>
_[arXiv'23]() |
[GitHub](https://github.com/Wuziyi616/EventCLIP) |
[arXiv]()_

## Introduction

This is the official PyTorch implementation for paper: [EventCLIP: Adapting CLIP for Event-based Object Recognition]().
The code contains:

-   Zero-shot EventCLIP inference on N-Caltech, N-Cars, N-ImageNet datasets
-   Few-shot EventCLIP training and testing on the 3 datasets, with state-of-the-art classification performance

## Update

-   2023.5.17: Initial code release!

## Installation

Please refer to [install.md](docs/install.md) for step-by-step guidance on how to install the packages.

## Experiments

**This codebase is tailored to [Slurm](https://slurm.schedmd.com/documentation.html) GPU clusters with preemption mechanism.**
For the configs, we mainly use A40 with 40GB memory (though many experiments don't require so much memory).
Please modify the code accordingly if you are using other hardware settings:

-   Please go through `train.py` and change the fields marked by `TODO:`
-   Please read the config file for the model you want to train.
    We use DDP with multiple GPUs to accelerate training.
    You can use less GPUs to achieve a better memory-speed trade-off

### Dataset Preparation

Please refer to [data.md](docs/data.md) for dataset downloading and pre-processing.

### Reproduce Results

Please see [benchmark.md](docs/benchmark.md) for detailed instructions on how to reproduce our results in the paper.

## Possible Issues

See the troubleshooting section of [nerv](https://github.com/Wuziyi616/nerv#possible-issues) for potential issues.

Please open an issue if you encounter any errors running the code.

## Citation

Please cite our paper if you find it useful in your research:

```

```

## Acknowledgement

We thank the authors of [CLIP](https://github.com/openai/CLIP), [EST](https://github.com/uzh-rpg/rpg_event_representation_learning), [n_imagenet](https://github.com/82magnolia/n_imagenet), [PointCLIP](https://github.com/ZrrSkywalker/PointCLIP) for opening source their wonderful works.

## License

EventCLIP is released under the MIT License. See the LICENSE file for more details.

## Contact

If you have any questions about the code, please contact Ziyi Wu dazitu616@gmail.com
