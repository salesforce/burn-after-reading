'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from torchvision import datasets, transforms
import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path
import cv2
import torchvision
from randaugment import RandAugmentMC
from wilds.common.data_loaders import get_eval_loader
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset

torch.manual_seed(999)
np.random.seed(999)
torch.backends.cudnn.benchmark = True


def load_training_from_list(root, list_path, batch_size, kwargs, shuffle=True, return_idx=False):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    txt = open(list_path).readlines()
    data = ImageList_idx(root, txt, transform=transform, return_idx=return_idx)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=True, **kwargs)
    return train_loader


def load_training_strong_weak(root, list_path, batch_size, kwargs, shuffle=True, return_idx=False,
                              return_test_img=False):
    txt = open(list_path).readlines()
    data = Imagelist_strong_weak(root, txt, return_idx=return_idx, return_test_img=return_test_img)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=True, **kwargs)
    return train_loader


def load_testing_from_list(root, list_path, batch_size, kwargs, shuffle=False, return_idx=False):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    txt = open(list_path).readlines()
    data = ImageList_idx(root, txt, transform=transform, return_idx=return_idx)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=False, **kwargs)
    return train_loader


def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class ImageList_idx(Dataset):
    def __init__(self, root, image_list, labels=None, transform=None, target_transform=None, mode='RGB',
                 return_idx=True,
                 idx_mask=None):
        imgs = make_dataset(image_list, labels)
        self.root = root
        self.imgs = imgs
        if idx_mask is not None:
            self.imgs = [imgs[i] for i in idx_mask]

        self.transform = transform
        self.target_transform = target_transform
        self.return_idx = return_idx

        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        path = os.path.join(self.root, path)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_idx:
            return img, target, index
        else:
            return img, target

    def __len__(self):
        return len(self.imgs)


def get_index_label(txt):
    image_list = open(txt).readlines()
    data = [(i, int(val.split()[1])) for i, val in enumerate(image_list)]
    return np.array(data)


class Imagelist_strong_weak(object):
    def __init__(self, root, image_list, return_idx=False, return_test_img=False):
        imgs = make_dataset(image_list, labels=None)
        self.root = root

        self.imgs = imgs
        self.loader = rgb_loader
        self.return_idx = return_idx
        self.return_test_img = return_test_img
        self.test = transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(size=224)])
        self.weak = transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224)])
        self.strong = transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path, target = self.imgs[index]
        path = os.path.join(self.root, path)
        img = self.loader(path)

        img_strong = self.normalize(self.strong(img))
        img_weak = self.normalize(self.weak(img))
        img_test = self.normalize(self.test(img))

        if not self.return_idx:
            if not self.return_test_img:
                return (img_weak, img_strong), target
            else:
                return (img_weak, img_strong, img_test), target
        else:
            if not self.return_test_img:
                return (img_weak, img_strong), target, index
            else:
                return (img_weak, img_strong, img_test), target, index

    def __len__(self):
        return len(self.imgs)
