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
from bohb.BOWorker import BOWorker

parser = argparse.ArgumentParser("kuzushiji")
parser.add_argument('--data_dir', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--weigthed_sample', action='store_true', default=False, help='Use weighted sampling')
parser.add_argument('--set', type=str, default='KMNIST', help='The dataset to be trained')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=20, help='num of training epochs')
parser.add_argument('--input_channels', type=int, default=1, help='num of input channels')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--log_dir', type=str, default='./log', help='logging file location')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='PCDARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=int, default=5, help='gradient clipping')
parser.add_argument('--min_budget', type=int, default=6, help='minimal budget')
parser.add_argument('--max_budge', type=int, default=20, help='minimal budget')
parser.add_argument('--n_iterations', type=int, default=12, help='number of iterations for BO optimizer')
parser.add_argument('--train_portion', type=float, default=0.8, help='portion of training data')

args = parser.parse_args()


# logging 
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.data_dir, exist_ok=True)

current_date = datetime.datetime.now()
timestamp = current_date.isoformat()

data_dir = args.data_dir
log_path = args.log_dir+'/exp_{}'.format(timestamp)
os.makedirs(log_path, exist_ok=True)
log_name = "log_bo_" + args.set + ".txt"
log_dir = os.path.join(log_path, log_name)

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

  server = '127.0.0.1'
  # start hpns server
  NS = hpns.NameServer(run_id='example1', host=server, port=None)
  NS.start()
  # BOHB worker
  worker = BOWorker(run_id='example1', train_func=train_func)
  worker.run(background=True)

  # BOHB optimizer
  bohb = BOHB(configspace=worker.get_configspace(),
              run_id='example1', nameserver=server,
              min_budget=args.min_budget, max_budget=args.max_budget
            )

  res = bohb.run(n_iterations=args.n_iterations)

  bohb.shutdown(shutdown_workers=True)
  NS.shutdown()

  result_file = os.path.join(self.logdir, 'bohb_result.pkl')

  with open(result_file, 'wb') as f:
    pickle.dump(res, f)


def train_func(config, budget):
  num_layers = config["num_layers"]
  model_learning_rate = config["model_learning_rate"]
  auxiliary_weight = config["auxiliary_weight"]
  cutout_length = config["cutout_length"]
  init_channel = config["init_channel"]
  drop_path_prob = config["drop_path_prob"]
  weight_decay = config["weight_decay"]

  model = Network(init_channel, args.input_channels, num_classes, num_layers, args.auxiliary, genotype)
  model = model.cuda()
  logging.info(config)
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.Adam(
      model.parameters(),
      model_learning_rate,
      weight_decay=weight_decay
      )

  # TODO: 
  transform_config = 

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

  # Train/Valid/Test split
  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
    pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
    pin_memory=True, num_workers=2)

  test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(budget))

  train_acc_lst, train_obj_lst = [], []

  for epoch in range(budget):
    scheduler.step()
    logging.info('epoch %d/%d (budget) lr %e', epoch+1, budget, scheduler.get_lr()[0])

    model.drop_path_prob = drop_path_prob * epoch / budget

    # TODO: 
    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    train_acc_lst.append(train_acc)
    train_obj_lst.append(train_obj)

    logging.info('(balanced)train_acc %f', train_acc)

  valid_acc, valid_obj = infer(valid_queue, model, criterion)
  
  logging.info('(balanced) valid_acc %f', valid_acc)
  
  return train_acc_lst, train_obj_lst, valid_acc, valid_obj


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
      loss += args.auxiliary_weight * loss_aux
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
