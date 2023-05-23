import os
import copy
import torch
import pandas as pd

def construct_subset(dataset, split):
    split = pd.read_csv(split)['img_path'].values
    root = dataset.root

    class_to_idx = dataset.class_to_idx

    # create targets
    targets = [class_to_idx[os.path.dirname(i)] for i in split]


    image_names = [os.path.join(root, j) for j in split]
    dataset_subset = copy.deepcopy(dataset)

    dataset_subset.samples = [j for j in zip(image_names, targets)]
    dataset_subset.imgs = dataset_subset.samples
    dataset_subset.targets = targets
    return dataset_subset


