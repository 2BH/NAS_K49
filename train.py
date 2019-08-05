import os
import sys
import time
import glob
import numpy as np
import torch
import tqdm
from utils.datasets import KMNIST, K49
import utils.utils as utils
import logging
import datetime
import argparse
import torch.nn as nn
import core.genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from models.model import NetworkKMNIST
from models.model import NetworkK49


parser = argparse.ArgumentParser("kmnist")
parser.add_argument('--data_dir', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--data_aug', type=str, default=None, help='Data Augmentation method')
parser.add_argument('--set', type=str, default='KMNIST', help='The dataset to be trained')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=30, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--input_channels', type=int, default=1, help='num of input channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=14, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--log_dir', type=str, default='./log', help='logging file location')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='PCDARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()


# logging 
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.data_dir, exist_ok=True)

timestamp = "2019-08-02T13:57:54.488278"

data_dir = args.data_dir
log_path = args.log_dir+'/exp_{}'.format(timestamp)
os.makedirs(log_path, exist_ok=True)
log_dir = os.path.join(log_path, 'log_test_aux_t.txt')

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

  genotype = eval("core.genotypes.%s" % args.arch)
  if args.set == "KMNIST":
    model = NetworkKMNIST(args.init_channels, args.input_channels, num_classes, args.layers, args.auxiliary, genotype)
  elif args.set == "K49":
    model = NetworkK49(args.init_channels, args.input_channels, num_classes, args.layers, args.auxiliary, genotype)

  model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  # Data augmentations
  data_augmentations = args.data_aug
  if data_augmentations is None:
    data_augmentations = transforms.ToTensor()
  elif isinstance(type(data_augmentations), list):
      data_augmentations = transforms.Compose(data_augmentations)
  elif not isinstance(data_augmentations, transforms.Compose):
      raise NotImplementedError
  
  # Dataset
  if args.set == "KMNIST":
    train_data = KMNIST(args.data_dir, True, data_augmentations)
    test_data = KMNIST(args.data_dir, False, data_augmentations)
  elif args.set == "K49":
    train_data = K49(args.data_dir, True, data_augmentations)
    test_data = K49(args.data_dir, False, data_augmentations)
  else:
    raise ValueError("Unknown Dataset %s" % args.dataset)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  best_acc = 0.0
  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d/%d lr %e', epoch, args.epochs, scheduler.get_lr()[0])

    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    if valid_acc > best_acc:
        best_acc = valid_acc
    logging.info('valid_acc %f, best_acc %f', valid_acc, best_acc)

    utils.save(model, os.path.join(log_path, 'weights.pt'))


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in tqdm.tqdm(enumerate(train_queue), disable=True):
    input = torch.tensor(input).cuda()
    target = torch.tensor(target).cuda(async=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    (prec1,) = utils.accuracy(logits, target, topk=(1,))
    n = input.size(0)
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
      input = torch.tensor(input).cuda()
      target = torch.tensor(target).cuda(async=True)

      logits, _ = model(input)
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

