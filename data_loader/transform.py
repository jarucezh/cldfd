# Performs well on three dataset, do not modify.
import torch
import torchvision
import torchvision.transforms as transforms



class TripleTrans:
    def __init__(self, transform1, transform2, transform3):
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3


    def __call__(self, x):
        return [self.transform1(x), self.transform2(x), self.transform3(x)]


class TransformLoader:
    def __init__(self, image_size,
                 normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):

        if transform_type == 'RandomColorJitter':
            return torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)], p=1.0)
        if transform_type == 'RandomGrayscale':
            return torchvision.transforms.RandomGrayscale(p=0.1)
        elif transform_type == 'RandomGaussianBlur':
            return torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(kernel_size=(5, 5))], p=0.3)
        elif transform_type == 'CenterCrop':
            return torchvision.transforms.CenterCrop(self.image_size)
        elif transform_type == 'Scale':
            return torchvision.transforms.Resize([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == 'Normalize':
            return torchvision.transforms.Normalize(**self.normalize_param)
        elif transform_type == 'RandomResizedCrop':
            return torchvision.transforms.RandomResizedCrop(self.image_size)
        elif transform_type == 'RandomCrop':
            return torchvision.transforms.RandomCrop(self.image_size)
        elif transform_type == 'Resize_up':
            return torchvision.transforms.Resize([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == 'Resize':
            return torchvision.transforms.Resize([int(self.image_size), int(self.image_size)])

        else:
            method = getattr(torchvision.transforms, transform_type)
            return method()

    def get_composed_transform(self, aug=False):
        if aug == "strong":
            transform_list = ['RandomResizedCrop', 'RandomColorJitter', 'RandomGrayscale', 'RandomGaussianBlur',
                              'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        elif aug == "weak":
            transform_list = ['RandomResizedCrop', 'RandomColorJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']

        else:
            transform_list = ['Resize_up', 'CenterCrop', 'ToTensor', 'Normalize']


        if aug in ("strong_strong_strong", "strong_strong_weak", "strong_weak_weak", "weak_weak_weak", "test_test_test"):
            augs = aug.split("_")
            tfms3 = {"transform1": self.get_composed_transform(augs[0]),
                    "transform2": self.get_composed_transform(augs[1]),
                    "transform3": self.get_composed_transform(augs[2])}
            transform = TripleTrans(**tfms3)
            return transform


        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform
