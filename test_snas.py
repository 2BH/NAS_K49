import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from snas.model_search import Network
from snas.model_search_cons import ConsNetwork
from snas.option.default_option import TestOptions
from utils.datasets import KMNIST, K49
from utils.utils import *
import torchvision.transforms as transforms
import os
import sys
import logging
import datetime
import tqdm
import warnings
import argparse

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # device = torch.device('cuda')
    opt = TestOptions()

    data_dir = args.data_dir

    # Data augmentations
    data_augmentations = args.data_aug
    if data_augmentations is None:
        # You can add any preprocessing/data augmentation you want here
        data_augmentations = transforms.ToTensor()
    elif isinstance(type(data_augmentations), list):
        data_augmentations = transforms.Compose(data_augmentations)
    elif not isinstance(data_augmentations, transforms.Compose):
        raise NotImplementedError

    # Dataset
    if args.dataset == "KMNIST":
        train_data = KMNIST(data_dir, True, data_augmentations)
        test_data = KMNIST(data_dir, False, data_augmentations)
    elif args.dataset == "K49":
        train_data = K49(data_dir, True, data_augmentations)
        test_data = K49(data_dir, False, data_augmentations)
    else:
        raise ValueError("Unknown Dataset %s" % args.dataset)

    num_classes = train_data.n_classes
    criterion = nn.CrossEntropyLoss().cuda()

    # Select model: (constraint or not)
    if args.constraint:
        model = ConsNetwork(opt.init_channels, opt.input_channels, num_classes, opt.layers, criterion)    
    else:
        model = Network(opt.init_channels, opt.input_channels, num_classes, opt.layers, criterion)

    model.cuda()

    optimizer_model = torch.optim.SGD(model.parameters(),
                                      lr = opt.learning_rate,
                                      momentum = opt.momentum,
                                      weight_decay=opt.weight_decay)
    optimizer_arch = torch.optim.Adam(model.arch_parameters(),
                                      lr = opt.arch_learning_rate,
                                      betas = opt.betas,
                                      weight_decay = opt.arch_weight_decay)

    # KMNIST
    num_train = len(train_data)
    num_train_data = num_train // 2
    num_valid_data = num_train - num_train_data

    indices = list(range(num_train))

    # Dataloader
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=opt.batch_size,
        # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:num_train_data]),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        test_data, batch_size=opt.batch_size,
        # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[num_train_data:num_train_data+num_valid_data]),
        pin_memory=True, num_workers=2)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_model, float(opt.epochs), eta_min=opt.learning_rate_min)

    # Stats
    lst_train_acc = []
    lst_test_acc = []

    model.load_state_dict(torch.load('log/exp_2019-08-01T15:03:20.786846/weights.pt'))

    alpha_normal, alpha_reduce = model.arch_parameters()

    print(model.arch_parameters)
    print("-------------------------------------------------")
    print(alpha_normal)

    print("-------------------------------------------------")
    print(alpha_reduce)
    # np.save("alpha_normal", alpha_normal.detach.numpy())
    # np.save("alpha_reduce", alpha_reduce.detach.numpy())

    """
    for epoch in range(opt.epochs):
        logging.info("Starting epoch %d/%d", epoch+1, opt.epochs)
        np.random.seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        # training
        train_acc, train_valoss, train_poloss = train(train_queue,
                                                      valid_queue,
                                                      model,
                                                      criterion,
                                                      optimizer_arch,
                                                      optimizer_model,
                                                      opt,
                                                      args)

        # validation
        valid_acc, valid_valoss = infer(valid_queue, model, criterion, opt)
        
        lst_train_acc.append(train_acc)
        lst_test_acc.append(test_acc)

        scheduler.step()
    """

def train(train_queue, model, criterion, optimizer_arch, optimizer_model, opt, args):
    objs = AvgrageMeter()
    policy  = AvgrageMeter()
    top1 = AvgrageMeter()

    for step, (input, target) in tqdm.tqdm(enumerate(train_queue)):
        model.train()
        n = input.size(0) # batch size 

        input = torch.tensor(input, requires_grad=True).cuda()
        target = torch.tensor(target).cuda(async=True)
        
        input_search, target_search = next(iter(valid_queue))
        input_search = torch.tensor(input_search, requires_grad=True).cuda()
        target_search = torch.tensor(target_search).cuda(async=True)
        
        temperature = opt.initial_temp * np.exp(-opt.anneal_rate * step)

        optimizer_arch.zero_grad()
        optimizer_model.zero_grad()
        
        if args.constraint:
            logit, _, cost = model(input, temperature)## model inputs 
            _, score_function, __ = model(input_search, temperature)## model inputs     
            policy_loss = torch.sum(score_function * model.Credit(input_search, target_search, temperature).float())
            value_loss = criterion(logit, target)
            total_loss = policy_loss + value_loss + cost*(1e-9)
        else:
            logit, _ = model(input, temperature)## model inputs 
            _, score_function = model(input_search, temperature)## model inputs 
            policy_loss = torch.sum(score_function * model.Credit(input_search, target_search, temperature).float())
            value_loss = criterion(logit, target)
            total_loss = policy_loss + value_loss

        total_loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 5)
        optimizer_arch.step()
        optimizer_model.step()

        (prec1,) = accuracy(logit, target, topk=(1,))
        objs.update(value_loss.data, n)
        policy.update(policy_loss.data , n)
        top1.update(prec1.data, n)
        
    return top1.avg, objs.avg, policy.avg


def infer(valid_queue, model, criterion, opt):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target) in tqdm.tqdm(enumerate(valid_queue)):
            input = torch.tensor(input).cuda()
            target = torch.tensor(target).cuda(async=True)

            temperature = opt.initial_temp * np.exp(-opt.anneal_rate * step)
            logits, _ = model(input, temperature)
            loss = criterion(logits, target)
            (prec1,) = accuracy(logits, target, topk=(1,))
            n = input.size(0)
            objs.update(loss.data , n)
            top1.update(prec1.data , n)

    return top1.avg, objs.avg


# Main
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default="./data", type=str, nargs='?', help='specifies the path to data')
parser.add_argument('--log_dir', default="./log", type=str, nargs='?', help='specifies the path to logging file')
parser.add_argument('--constraint', default=False, action='store_true', help='If True, use constrainted model')
parser.add_argument('--dataset', default="KMNIST", type=str, nargs='?', help='specifies the dataset to use')
parser.add_argument('--data_aug', default=None, type=str, nargs='?', help='specifies the data augmentation to use')
args = parser.parse_args()

main(args)
