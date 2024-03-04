import logging
import math

import torch
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(args, root):

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)])

    base_dataset = datasets.CIFAR10(
        root, train=True, download=True)

    train_uni_dataset = UNI_CIFAR10(
        root, train=True,
        transform=Transform_data(mean=cifar100_mean, std=cifar100_std),
        ratio_labeled=args.rl,
        ratio_partiallabeled=args.rp,
        ratio_unlabeled=args.ru,
        partial_type=args.partial_type,
        partial_rate=args.partial_rate)

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_uni_dataset, test_dataset


def get_cifar100(args, root):

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_uni_dataset = UNI_CIFAR100(
        root, train=True,
        transform=Transform_data(mean=cifar100_mean, std=cifar100_std),
        ratio_labeled=args.rl,
        ratio_partiallabeled=args.rp,
        ratio_unlabeled=args.ru,
        partial_type=args.partial_type,
        partial_rate=args.partial_rate)

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_uni_dataset, test_dataset


def labels_to_one_hot(labels, num_classes=100):
    # Convert the list of labels to a tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Create a one-hot encoded tensor of shape (len(labels), num_classes)
    one_hot_labels = torch.zeros(len(labels), num_classes, dtype=torch.float32)
    one_hot_labels.scatter_(1, labels_tensor.unsqueeze(1), 1.0)

    return one_hot_labels


def partialize(y, y0, t, p):
    new_y = y.clone()
    n, c = y.shape[0], y.shape[1]
    avgC = 0

    if t == 'binomial':
        for i in range(n):
            row = new_y[i, :]
            row[np.where(np.random.binomial(1, p, c) == 1)] = 1
            while torch.sum(row) == 1:
                row[np.random.randint(0, c)] = 1
            avgC += torch.sum(row)
            new_y[i] = row / torch.sum(row)

    if t == 'pair':
        P = np.eye(c)
        for idx in range(0, c-1):
            P[idx, idx], P[idx, idx+1] = 1, p
        P[c-1, c-1], P[c-1, 0] = 1, p
        for i in range(n):
            row = new_y[i, :]
            idx = y0[i]
            row[np.where(np.random.binomial(1, P[idx, :], c)==1)] = 1
            avgC += torch.sum(row)
            new_y[i] = row / torch.sum(row)

    avgC = avgC / n
    return new_y, avgC



def x_partition(num_classes, num_chosen, labels, ex_idx):
    num_per_class = num_chosen // num_classes
    num_rest = num_chosen % num_classes        # To prevent un-integer division


    labels = np.array(labels)
    labels[ex_idx] = -1
    chosen_idx = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, num_per_class, False)
        chosen_idx.extend(idx)

    labels[chosen_idx] = -1
    idx = np.where(labels != -1)[0]
    idx = np.random.choice(idx, num_rest, False)
    chosen_idx.extend(idx)
    chosen_idx = np.array(chosen_idx)

    assert len(chosen_idx) == num_chosen

    np.random.shuffle(chosen_idx)
    return chosen_idx



    
class Transform_data(object):
    def __init__(self, mean, std):

        self.origin = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

        self.weak = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        self.strong = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(3, 5),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        self.strong1 = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(3, 5),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    def __call__(self, x):
        return self.weak(x), self.strong(x), self.strong1(x)


class UNI_CIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,download=False,
                 ratio_labeled=0.2,
                 ratio_partiallabeled=0.4,
                 ratio_unlabeled=0.4,
                 partial_type='binomial',
                 partial_rate=0.1):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets_onehot = labels_to_one_hot(self.targets,10)
        self.masks = torch.zeros_like(self.targets_onehot)

        self.nl = int(ratio_labeled * len(self.data))
        self.np = int(ratio_partiallabeled * len(self.data))
        self.nu = int(ratio_unlabeled * len(self.data))

        labeled_idx = x_partition(100, self.nl, self.targets, [])
        partiallabeled_idx = x_partition(100, self.np, self.targets, labeled_idx)
        unlabeled_idx = x_partition(100, self.nu, self.targets,list(set(partiallabeled_idx) | set(labeled_idx)))

        # labeled data
        self.masks[:self.nl] = self.targets_onehot[:self.nl]
        # partial-labeled data
        if (self.nl != 50000):
            self.targets_onehot[self.nl:self.nl+self.np], _ = partialize(self.targets_onehot[self.nl:self.nl+self.np],
                                                self.targets[self.nl:self.nl+self.np], partial_type, partial_rate)
            self.masks[self.nl:self.nl+self.np] = (self.targets_onehot[self.nl:self.nl+self.np] > 0)
        # unlabeled data
            self.targets_onehot[self.nl + self.np : self.nl + self.np + self.nu] = torch.ones(self.nu, 10) / 10

    def update_psdlabel(self, inst_psdlabel, idx, alpha):
        self.targets_onehot[idx] = self.targets_onehot[idx] * alpha + inst_psdlabel * (1 - alpha)

    def __getitem__(self, index):
        img, target, target_onehot, mask = self.data[index], self.targets[index], \
                                           self.targets_onehot[index], self.masks[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target_onehot = self.target_transform(target_onehot)

        return img, target_onehot, mask, index


class UNI_CIFAR100(datasets.CIFAR100):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,download=False,
                 ratio_labeled=0.2,
                 ratio_partiallabeled=0.4,
                 ratio_unlabeled=0.4,
                 partial_type='binomial',
                 partial_rate=0.1):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        print(ratio_labeled, ratio_partiallabeled, ratio_unlabeled)
        self.targets_onehot = labels_to_one_hot(self.targets,100)
        self.masks = torch.zeros_like(self.targets_onehot)

        self.nl = int(ratio_labeled * len(self.data))           # math.floor the last one using total minus to ensure == total
        self.np = int(ratio_partiallabeled * len(self.data))
        self.nu = int(ratio_unlabeled * len(self.data))

        l_idx = x_partition(100, self.nl, self.targets, [])
        p_idx = x_partition(100, self.np, self.targets, l_idx)
        u_idx = x_partition(100, self.nu, self.targets,list(set(p_idx) | set(l_idx)))

            
        # labeled data
        self.masks[l_idx] = self.targets_onehot[l_idx]
        # partial-labeled data
        if(p_idx != []):
            self.targets_onehot[p_idx], _ = partialize(self.targets_onehot[p_idx],
                                                self.targets[p_idx], partial_type, partial_rate)
            self.masks[p_idx] = (self.targets_onehot[p_idx] > 0)
        # unlabeled data
        self.targets_onehot[u_idx] = torch.ones(self.nu, 100) / 100

    def update_psdlabel(self, inst_psdlabel, idx, alpha):
        self.targets_onehot[idx] = self.targets_onehot[idx] * alpha + inst_psdlabel * (1 - alpha)

    def __getitem__(self, index):
        img, target, target_onehot, mask = self.data[index], self.targets[index], \
                                           self.targets_onehot[index], self.masks[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target_onehot = self.target_transform(target_onehot)

        return img, target_onehot, mask, index


DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100}
