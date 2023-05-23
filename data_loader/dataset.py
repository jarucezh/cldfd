import os

import numpy as np
import torch
import pickle
import pandas as pd
from .helper import *
from PIL import Image
from torch.utils.data import Dataset

import torchvision.datasets as datasets



data_root = os.path.expanduser("/home")
miniimagenet = {"train": '/gs/hs0/tga-aklab/bruce/tseng_l3_onecycle/filelists/miniImagenet/source/mini_imagenet_full_size/train'}
eurosat = "/gs/hs0/tga-aklab/bruce/cdfsl/2750/"
cropdisease = "/gs/hs0/tga-aklab/bruce/cdfsl/CropDiseases/dataset/train/"
chestx = "/gs/hs0/tga-aklab/bruce/cdfsl/chestX"
isic = "/gs/hs0/tga-aklab/bruce/cdfsl/isic"


identity_transform = lambda x: x

class TorchDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, index):
        image, label = self.dset[index]
        image = image.convert("RGB")
        return image, label

class MiniImagenet:
    def __init__(self, transform, target_transform = identity_transform, mode = "train", split = None):
        self.d = datasets.ImageFolder(miniimagenet[mode],
                                      transform=transform,
                                      target_transform=target_transform)
    def __getitem__(self, index):
        return self.d[index]

    def __len__(self):
        return len(self.d)



class EuroSAT:
    def __init__(self, transform, target_transform = identity_transform, split = None):
        self.d = datasets.ImageFolder(eurosat,
                                      transform=transform,
                                      target_transform=target_transform)

        if split is not None:
            self.d = construct_subset(self.d, split)


    def __getitem__(self, index):
        return self.d[index]

    def __len__(self):
        return len(self.d)


class CropDisease:
    def __init__(self, transform, target_transform = identity_transform, split = None):
        self.d = datasets.ImageFolder(cropdisease,
                                      transform=transform,
                                      target_transform=target_transform)

        if split is not None:
            self.d = construct_subset(self.d, split)


    def __getitem__(self, index):
        return self.d[index]

    def __len__(self):
        return len(self.d)

class CUB:
    def __init__(self, transform, target_transform = identity_transform, split = None):
        self.d = datasets.ImageFolder(cropdisease,
                                      transform=transform,
                                      target_transform=target_transform)

        if split is not None:
            self.d = construct_subset(self.d, split)


    def __getitem__(self, index):
        return self.d[index]

    def __len__(self):
        return len(self.d)


class ChestX(Dataset):
    def __init__(self, transform, target_transform = identity_transform, split = None,
                 csv_path = chestx + "/Data_Entry_2017.csv",
                 image_path = chestx + "/images/"):

        self.transform = transform
        self.target_transform = target_transform

        self.image_path = image_path
        self.csv_path = csv_path
        self.used_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
                            "Mass","Nodule", "Pneumonia", "Pneumothorax"]
        self.labels_maps = {"Atelectasis": 0,"Cardiomegaly": 1,"Effusion": 2,"Infiltration": 3,
                            "Mass": 4,"Nodule": 5,"Pneumothorax": 6}


        self.data_info = pd.read_csv(csv_path, skiprows=[0], header = None)

        self.images_name_all = np.asarray(self.data_info.iloc[:, 0])
        self.label_all = np.asarray(self.data_info.iloc[:, 1])

        self.image_name, self.labels = [], []



        for name, label in zip(self.images_name_all, self.label_all):
            label = label.split("|")

            if len(label) == 1 and label[0] != "No Finding" and label[0] != "Pneumonia" and label[
                0] in self.used_labels:

                self.labels.append(self.labels_maps[label[0]])
                self.image_name.append(name)

        self.data_len = len(self.image_name)
        self.image_name = np.asarray(self.image_name)
        self.labels = np.asarray(self.labels)

        if split is not None:
            split = pd.read_csv(split)['img_path'].values
            ind = np.concatenate([np.where(self.image_name == j)[0] for j in split])
            self.image_name = self.image_name[ind]
            self.labels = self.labels[ind]
            self.data_len = len(split)

            assert len(self.image_name) == len(split)
            assert len(self.labels) == len(split)

    def __getitem__(self, index):

        single_name = self.image_name[index]
        image = Image.open(self.image_path + single_name).resize((256, 256)).convert("RGB")
        image.load()

        label = self.labels[index]
        return self.transform(image), self.target_transform(label)


    def __len__(self):
        return self.data_len


class ISIC(Dataset):
    def __init__(self, transform, target_transform=identity_transform,
                 csv_path=isic + "/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv", \
                 image_path=isic + "/ISIC2018_Task3_Training_Input/", split=None):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
            target_transform: pytorch transforms for targets
            split: the filename of a csv containing a split for the data to be used.
                    If None, then the full dataset is used. (Default: None)
        """
        self.img_path = image_path
        self.csv_path = csv_path

        # Transforms
        self.transform = transform
        self.target_transform = target_transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name = np.asarray(self.data_info.iloc[:, 0])

        self.labels = np.asarray(self.data_info.iloc[:, 1:])
        self.labels = (self.labels != 0).argmax(axis=1)

        # Calculate len
        self.data_len = len(self.image_name)
        self.split = split

        if split is not None:
            print("Using Split: ", split)
            split = pd.read_csv(split)['img_path'].values
            # construct the index
            ind = np.concatenate([np.where(self.image_name == j)[0] for j in split])
            self.image_name = self.image_name[ind]
            self.labels = self.labels[ind]
            self.data_len = len(split)

            assert len(self.image_name) == len(split)
            assert len(self.labels) == len(split)
        # self.targets = self.labels

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_name[index]
        # Open image
        temp = Image.open(self.img_path + single_image_name + ".jpg")
        img_as_img = temp.copy()
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.labels[index]

        return self.transform(img_as_img), self.target_transform(single_image_label)

    def __len__(self):
        return self.data_len
