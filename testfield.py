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
train_data = KMNIST(data_dir, True, data_augmentations)
test_data = KMNIST(data_dir, False, data_augmentations)

# print(train_data.class_frequency)
# print(test_data.class_frequency)
#images = np.expand_dims(train_data.images, axis=-1)
#print(images.shape)
#trans = transforms.ToTensor()
#images2 = trans(images)
#print(images2.size())
print(np.mean(train_data.images.reshape(-1) / 255))
print(np.std(train_data.images.reshape(-1) / 255))

"""
out = torch.tensor([[0.8, 0.1, 0.1],
                    [0.2, 0.7, 0.1],
                    [0.1, 0.1, 0.8],
                    [0.2, 0.1, 0.7]])
tar = torch.tensor([0, 1, 1, 2]) # uncorrect on majority classe
tar2 = torch.tensor([1, 1, 2, 2]) # uncorrect on minority class
tar3 = torch.tensor([0, 1, 2, 2]) # all correct

def balanced_accuracy(output, target, class_weight):
  batch_size = target.size(0)
  pred = output.argmax(dim=1)

  pred = pred.t()

  correct = pred.eq(target.view(1, -1).expand_as(pred))
"""
