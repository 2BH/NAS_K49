import matplotlib.pyplot as pyplot
import numpy as np
import torch
from utils.datasets import * 
from torch.utils.data import DataLoader


data_dir = "./data"
data_augmentations = None
train_dataset = K49(data_dir, True, data_augmentations)
test_dataset = K49(data_dir, False, data_augmentations)

num_train = len(train_dataset)
indices = list(range(num_train))


valid_queue = torch.utils.data.DataLoader(
    train_dataset, batch_size=16,
    #sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:50000]),
    pin_memory=True, num_workers=2)

"""
train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False)
"""
input_search, target_search = next(iter(valid_queue))

print(input_search.shape)