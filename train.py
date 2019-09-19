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
from models.model import KuzushijiNet as Network 
from sklearn.metrics import balanced_accuracy_score


parser = argparse.ArgumentParser("kuzushiji")
parser.add_argument('--data_dir', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='Use weighted sampling')
parser.add_argument('--set', type=str, default='KMNIST', help='The dataset to be trained')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=20, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--input_channels', type=int, default=1, help='num of input channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--optimizer', type=str, default='Adam', help='Type of optimizer to use')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=6, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--log_dir', type=str, default='./log', help='logging file location')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='PCDARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()


# logging 
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.data_dir, exist_ok=True)

current_date = datetime.datetime.now()
timestamp = current_date.isoformat()

data_dir = args.data_dir
log_path = args.log_dir+'/exp_{}'.format(timestamp)
os.makedirs(log_path, exist_ok=True)
log_name = "log_test_" + args.set + ".txt"
log_dir = os.path.join(log_path, 'log_test_aux.txt')

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

  model = Network(args.init_channels, args.input_channels, num_classes, args.layers, args.auxiliary, genotype)

  model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  if args.optimizer == "Adam":
    optimizer = torch.optim.Adam(
      model.parameters(),
      args.learning_rate,
      # momentum=args.momentum,
      weight_decay=args.weight_decay
      )
  else:
    optimizer = torch.optim.AdamW(
      model.parameters(),
      args.learning_rate,
      # momentum=args.momentum,
      weight_decay=args.weight_decay
      )
  # Data augmentations
  train_transform, valid_transform = utils.data_transforms_Kuzushiji(args)
  
  # Dataset
  if args.set == "KMNIST":
    train_data = KMNIST(args.data_dir, True, train_transform)
    test_data = KMNIST(args.data_dir, False, valid_transform)
  elif args.set == "K49":
    train_data = K49(args.data_dir, True, train_transform)
    test_data = K49(args.data_dir, False, valid_transform)
  else:
    raise ValueError("Unknown Dataset %s" % args.dataset)

  if args.weighted_sample and args.set == "K49":
    # enable weighted sampler

    train_weights = 1 / train_data.class_frequency[train_data.labels]
    train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.WeightedRandomSampler(weights=train_weights,
                                                             num_samples=len(train_weights)),
      pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size,
      pin_memory=True, num_workers=2)
  else:
    train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True,
      pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size,
      pin_memory=True, num_workers=2)
  
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, float(args.epochs), eta_min=args.learning_rate_min)
  best_acc = 0.0
  genotype = eval("core.genotypes.%s" % args.arch)
  print('---------Genotype---------')
  logging.info(genotype)
  print('--------------------------')
  train_acc_lst = []
  valid_acc_lst = []
  for epoch in range(args.epochs):

    logging.info('epoch %d/%d lr %e', epoch+1, args.epochs, scheduler.get_lr()[0])


    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)
    train_acc_lst.append(train_acc)
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    valid_acc_lst.append(valid_acc)
    is_best = False
    if valid_acc > best_acc:
        best_acc = valid_acc
        is_best = True
    logging.info('(unbalanced) valid_acc %f, best_acc %f', valid_acc, best_acc)
    scheduler.step()
    utils.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_acc_top1': best_acc,
        'optimizer' : optimizer.state_dict(),
        }, is_best, log_path)
        
  logging.info("train acc")
  logging.info(train_acc_lst)
  logging.info("valid acc")
  logging.info(valid_acc_lst)

  # Dataset
  if args.set == "KMNIST":
    train_data = KMNIST(args.data_dir, True, valid_transform)
    test_data = KMNIST(args.data_dir, False, valid_transform)
  elif args.set == "K49":
    train_data = K49(args.data_dir, True, valid_transform)
    test_data = K49(args.data_dir, False, valid_transform)
  else:
    raise ValueError("Unknown Dataset %s" % args.dataset)
  
  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=False,
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size, shuffle=False,
      pin_memory=True, num_workers=2)
  

  model.eval()
  train_prediction = []  
  for step, (input, target) in tqdm.tqdm(enumerate(train_queue), disable=True):
    input = input.cuda()
    target = target.cuda()
    logits = model(input)[0]
    pred = logits.argmax(dim=-1)
    train_prediction.extend(pred.tolist())

  test_prediction = []
  for step, (input, target) in tqdm.tqdm(enumerate(valid_queue), disable=True):
    input = input.cuda()
    target = target.cuda()
    logits = model(input)[0]
    pred = logits.argmax(dim=-1)
    test_prediction.extend(pred.tolist())

  train_labels = train_data.labels
  test_labels = test_data.labels

  train_acc = np.sum(train_labels == train_prediction) / train_labels.shape[0]
  test_acc = np.sum(test_labels == test_prediction) / test_labels.shape[0]
  train_acc_bal = balanced_accuracy_score(train_labels, np.array(train_prediction))
  test_acc_bal = balanced_accuracy_score(test_labels, np.array(test_prediction))

  logging.info("(unbalanced) train acc: %f, valid acc: %f", train_acc, test_acc)
  logging.info("(balanced) train acc: %f, valid acc: %f", train_acc_bal, test_acc_bal)


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in tqdm.tqdm(enumerate(train_queue), disable=True):
    input = input.cuda()
    target = target.cuda()

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight * loss_aux
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    (prec1,) = utils.accuracy(logits, target, topk=(1,))
    n = input.size(0)
    objs.update(loss.data, n)
    top1.update(prec1.data, n)

    #if step % args.report_freq == 0:
    #  logging.info('train %03d %e %f', step, objs.avg, top1.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  model.eval()
  with torch.no_grad():
    for step, (input, target) in tqdm.tqdm(enumerate(valid_queue), disable=True):
      input = input.cuda()
      target = target.cuda()

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

