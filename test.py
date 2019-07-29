import matplotlib.pyplot as pyplot
import numpy as np
import torch
from utils.datasets import * 
from torch.utils.data import DataLoader


data_dir = "./data"
data_augmentations = None
train_dataset = K49(data_dir, True, data_augmentations)
test_dataset = K49(data_dir, False, data_augmentations)

train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False)