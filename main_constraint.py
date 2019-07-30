import numpy as np
import torch 
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
from torch.autograd import Variable
from snas.model_search_cons import Network
from snas.option.default_option import TrainOptions
from utils.datasets import *
from utils.utils import *
import os 
import warnings
warnings.filterwarnings("ignore")
import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda')
opt = TrainOptions()

data_dir = "./data/"
data_augmentations = None
if data_augmentations is None:
    # You can add any preprocessing/data augmentation you want here
    data_augmentations = transforms.ToTensor()
elif isinstance(type(data_augmentations), list):
    data_augmentations = transforms.Compose(data_augmentations)
elif not isinstance(data_augmentations, transforms.Compose):
    raise NotImplementedError

train_data = KMNIST(data_dir, True, data_augmentations)
test_data = KMNIST(data_dir, False, data_augmentations)

num_classes = train_data.n_classes
criterion = nn.CrossEntropyLoss().cuda()
model = Network(opt.init_channels, num_classes, opt.layers, criterion)
model.cuda()
optimizer_model = torch.optim.SGD(model.parameters(),lr= 0.025,momentum = 0.9, weight_decay=3e-4)
optimizer_arch = torch.optim.Adam(model.arch_parameters(),lr = 3e-4, betas=(0.5, 0.999), weight_decay = 1e-3)

num_train = len(train_data)
indices = list(range(num_train))
split = int(np.floor(opt.train_portion * num_train))

####DATALOADER
train_queue = torch.utils.data.DataLoader(
  train_data, batch_size=12,
  sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:5000]),
  pin_memory=True, num_workers=2)

valid_queue = torch.utils.data.DataLoader(
  train_data, batch_size=12,
  sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[10000:15000]),
  pin_memory=True, num_workers=4)



f = open("Loss.txt","w")

def train(train_queue,valid_queue, model, criterion, optimizer_arch, optimizer_model,lr_arch,lr_model):
    objs = AvgrageMeter()
    policy  = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    for step, (input, target) in tqdm.tqdm(enumerate(train_queue)):
        model.train()
        n = input.size(0) # batch size 

        input = torch.tensor(input , requires_grad = True).cuda()
        target = torch.tensor(target, requires_grad = False).cuda(async=True)
        
        input_search, target_search = next(iter(valid_queue))
        input_search = torch.tensor(input_search , requires_grad = True ).cuda()
        target_search = torch.tensor(target_search, requires_grad = False).cuda(async=True)
        
        temperature = opt.initial_temp * np.exp(-opt.anneal_rate * step)

        optimizer_arch.zero_grad()
        optimizer_model.zero_grad()
        logit , _ ,cost= model(input , temperature)## model inputs 
        _ , score_function ,__= model(input_search , temperature)## model inputs 
        
        policy_loss = torch.sum(score_function * model.Credit(input_search,target_search,temperature).float())
        value_loss = criterion(logit , target) 
        total_loss = policy_loss + value_loss + cost*(1e-9)
        total_loss.backward()
        nn.utils.clip_grad_norm(model.parameters(),5)
        optimizer_arch.step()
        optimizer_model.step()

        (prec1,) = accuracy(logit, target, topk=(1,))
        objs.update(value_loss.data, n)
        policy.update(policy_loss.data , n)
        top1.update(prec1.data , n)
    return top1.avg, objs.avg, policy.avg


def infer(valid_queue, model, criterion):
  objs = AvgrageMeter()
  top1 = AvgrageMeter()
  top5 = AvgrageMeter()
  model.eval()

  for step, (input, target) in tqdm.tqdm(enumerate(valid_queue)):
    input = torch.tensor(input, volatile=True).cuda()
    target = torch.tensor(target, volatile=True).cuda(async=True)

    temperature = opt.initial_temp * np.exp(-opt.anneal_rate * step)
    logits , _ , cost = model(input , temperature)
    loss = criterion(logits, target)
    (prec1,) = accuracy(logits, target, topk=(1,))
    n = input.size(0)
    objs.update(loss.data , n)
    top1.update(prec1.data , n)


  return top1.avg, objs.avg

####MAIN 

for epoch in range(opt.epochs):

    # training
    train_acc_top1 , train_valoss,train_poloss  = train(train_queue, valid_queue, model,criterion, optimizer_arch,optimizer_model, 3e-4,0.025)

    # validation
    valid_acc_top1, valid_valoss = infer(valid_queue, model, criterion)


    f.write("%5.5f  "% train_acc_top1)
    f.write("%5.5f  "% train_acc_top5)
    f.write("%5.5f  "% train_valoss)
    f.write("%5.5f  "% train_poloss ) 
    f.write("%5.5f  "% valid_acc_top1 ) 
    f.write("%5.5f  "% valid_acc_top5 ) 
    f.write("%5.5f  "% valid_valoss ) 
    f.write("\n")


    if epoch % 5 ==0:
      np.save("alpha_normal_" + str(epoch) + ".npy"  , model.alphas_normal.detach().cpu().numpy())
      np.save("alpha_reduce_" + str(epoch) + ".npy"  , model.alphas_reduce.detach().cpu().numpy())


    print("epoch : " , epoch , "Train_Acc : " , train_acc_top1 , "Train_value_loss : ",train_valoss,"Train_policy : " , train_poloss )
    print('\n')
    print("epoch : " , epoch , "Val_Acc : " , valid_acc_top1 , "Val_value_loss : ",valid_valoss)
    torch.save(model.state_dict(),'weights.pt')
f.close()






