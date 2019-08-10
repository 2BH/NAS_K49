from utils.datasets import K49, KMNIST
import torchvision.transforms as transforms
from sklearn.metrics import balanced_accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


data_dir = './data'
data_augmentations = None
if data_augmentations is None:
    data_augmentations = transforms.ToTensor()
elif isinstance(type(data_augmentations), list):
    data_augmentations = transforms.Compose(data_augmentations)
elif not isinstance(data_augmentations, transforms.Compose):
    raise NotImplementedError
train_data = K49(data_dir, True, data_augmentations)
test_data = K49(data_dir, False, data_augmentations)


# Generate the weights for sampler
train_frequency = train_data.class_frequency[train_data.labels]

print(train_frequency.shape)