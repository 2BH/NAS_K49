import os
import sys
import time
import numpy as np
import tqdm
import torch
import utils.utils as utils
import logging
import datetime
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from utils.datasets import KMNIST, K49

from models.model_search import Network
from core.architect import Architect


parser = argparse.ArgumentParser("kuzushiji")
parser.add_argument('--data_dir', type=str, default='./data', help='location of the data corpus')
# parser.add_argument('--data_aug', type=str, default=None, help='Data Augmentation method')
parser.add_argument('--set', type=str, default='KMNIST', help='The dataset to be trained')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--input_channels', type=int, default=1, help='num of input channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=6, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--log_dir', type=str, default='./log', help='logging file location')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.data_dir, exist_ok=True)

current_date = datetime.datetime.now()
timestamp = current_date.isoformat()
data_dir = args.data_dir
log_path = args.log_dir+'/exp_{}'.format(timestamp)
os.mkdir(log_path)
log_dir = os.path.join(log_path, 'log.txt')

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(log_dir)
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


if args.set == 'KMNIST':
  num_classes = 10
elif args.set == 'K49':
  num_classes = 49
else:
  raise ValueError("Invalid dataset name %s" % args.set)

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, args.input_channels, num_classes, args.layers, criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  # Data augmentations
  train_transform, _ = utils.data_transforms_Kuzushiji(args)
  # Dataset
  if args.set == "KMNIST":
    train_data = KMNIST(args.data_dir, True, train_transform)
  elif args.set == "K49":
    train_data = K49(args.data_dir, True, train_transform)
  else:
    raise ValueError("Unknown Dataset %s" % args.dataset)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
    pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
    pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  # counting time
  t = 0
  for epoch in range(args.epochs):
    t1 = time.time()
    
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d/%d lr %e', epoch, args.epochs, lr)

    # print the genotype
    genotype = model.genotype()
    logging.info('genotype = %s', genotype)
    
    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch)
    t2 = time.time()
    t += t2 - t1
    logging.info('train_acc %f', train_acc)

    # validation
    if args.epochs-epoch<=1:
      valid_acc, valid_obj = infer(valid_queue, model, criterion)
      logging.info('valid_acc %f', valid_acc)
    
    utils.save(model, os.path.join(log_path, 'weights.pt'))
  t = t/60/60
  logging.info("Training time cost: %f" % t)


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()

  for step, (input, target) in tqdm.tqdm(enumerate(train_queue), disable=True):
    model.train()
    n = input.size(0)
    input = torch.tensor(input, requires_grad=False).cuda()
    target = torch.tensor(target, requires_grad=False).cuda(async=True)

    # get a random minibatch from the search queue with replacement
    # input_search, target_search = next(iter(valid_queue))
    
    try:
      input_search, target_search = next(valid_queue_iter)
    except:
      valid_queue_iter = iter(valid_queue)
      input_search, target_search = next(valid_queue_iter)
    
    input_search = torch.tensor(input_search, requires_grad=False).cuda()
    target_search = torch.tensor(target_search, requires_grad=False).cuda(async=True)

    if epoch >= 10:
      architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    (prec1,) = utils.accuracy(logits, target, topk=(1,))
    objs.update(loss.data, n)
    top1.update(prec1.data, n)
    
    if step % args.report_freq == 0:
      logging.info('train %03d %e %f', step, objs.avg, top1.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  model.eval()
  with torch.no_grad():
    for step, (input, target) in tqdm.tqdm(enumerate(valid_queue), disable=True):
      #input = input.cuda()
      #target = target.cuda(non_blocking=True)
      input = torch.tensor(input, volatile=True).cuda()
      target = torch.tensor(target, volatile=True).cuda(async=True)
      logits = model(input)
      loss = criterion(logits, target)

      (prec1,) = utils.accuracy(logits, target, topk=(1,))
      n = input.size(0)
      objs.update(loss.data, n)
      top1.update(prec1.data, n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f', step, objs.avg, top1.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

