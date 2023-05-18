# Dataset Preparation

All datasets should be downloaded or soft-linked to `./data/`.
Or you can modify the `data_root` value in the config files.

## N-Caltech101

We adopt the N-Caltech101 dataset from [EST repo](https://github.com/uzh-rpg/rpg_event_representation_learning#training).
Please download and unzip the data and put it under `./data/N-Caltech101`.

## N-Cars

Please follow the instructions on the [official website](https://www.prophesee.ai/2018/03/13/dataset-n-cars/) to download N-Cars dataset and put it under `./data/N-Cars`.

## N-ImageNet

Please follow the instructions on the [n_imagenet repo](https://github.com/82magnolia/n_imagenet#n-imagenet-towards-robust-fine-grained-object-recognition-with-event-cameras) to download N-ImageNet and put it under `./data/N_Imagenet`.

The `N-ImageNet Variants (~150GB)` are not required for training.
But if you want to test the robustness of our method, you can download them as we do provide evaluation code on them.

The `Mini N-ImageNet (~45 GB)` is not used in this project.

## Summary

**The final `data` directory should look like this:**

```
data/
├── N-Caltech101/
│   ├── training/
│   │   ├── accordion/  # folder with events in '.npy' format
│   │   ├── airplanes/
•   •   •
•   •   •
│   │   └── yin_yang/
│   ├── validation/  # same as 'training'
│   ├── testing/  # same as 'training'
├── N-Cars/
│   ├── train/
│   │   ├── background/  # folder with events in '.npy' format
│   │   ├── cars/
│   ├── test/  # same as 'train'
├── N_Imagenet/
│   ├── extracted_train/
│   │   ├── n01440764/  # folder with events in '.npy' format
│   │   ├── n01443537/
•   •   •
•   •   •
│   │   └── n15075141/
│   ├── extracted_val/  # same as 'extracted_train'
│   ├── extracted_val_brightness_4/  # these are robustness variants
│   ├── extracted_val_brightness_5/  # brightness change
│   ├── extracted_val_brightness_6/
│   ├── extracted_val_brightness_7/
│   ├── extracted_val_mode_1/  # trajectory change
│   ├── extracted_val_mode_3/
│   ├── extracted_val_mode_5/
│   ├── extracted_val_mode_6/
└   └── extracted_val_mode_7/
```
