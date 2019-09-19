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
parser.add_argument('--set', type=str, default='KMNIST', help='The dataset to be trained')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=24, help='num of init channels')
parser.add_argument('--input_channels', type=int, default=1, help='num of input channels')
parser.add_argument('--layers', type=int, default=10, help='total number of layers')
parser.add_argument('--log_dir', type=str, default='./log', help='logging file location')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='PCDARTS', help='which architecture to use')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=6, help='cutout length')
args = parser.parse_args()


# logging 
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.data_dir, exist_ok=True)

timestamp = "2019-09-12T10:10:20.365069"
#2019-09-10T11:24:57.356339 K49_50_0
#exp_2019-09-10T11:26:13.550827
#exp_2019-09-10T11:26:13.551539
#exp_2019-09-10T12:51:49.163722
#exp_2019-09-10T12:51:49.164218
#exp_2019-09-10T12:51:49.164601

#exp_2019-09-12T10:10:20.364848
#exp_2019-09-12T10:10:20.364970
#exp_2019-09-12T10:10:20.365069

data_dir = args.data_dir
log_path = args.log_dir+'/exp_{}'.format(timestamp)
os.makedirs(log_path, exist_ok=True)
log_dir = os.path.join(log_path, 'log_test.txt')

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

  model = Network(args.init_channels, args.input_channels, num_classes, args.layers, True, genotype)
  # model = nn.DataParallel(model)
  model = model.cuda()

  # load model
  checkpoint = torch.load(log_path+'/checkpoint.pth.tar')
  model.load_state_dict(checkpoint['state_dict'])

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  # Data augmentations
  _, valid_transform = utils.data_transforms_Kuzushiji(args)
  
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
      train_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)


  genotype = eval("core.genotypes.%s" % args.arch)
  print('---------Genotype---------')
  logging.info(genotype)
  print('--------------------------') 

  model.drop_path_prob = 0.
  model.eval()
  train_prediction = []
  test_prediction = []
  with torch.no_grad():  
    for step, (input, target) in tqdm.tqdm(enumerate(train_queue), disable=True):
      input = input.cuda()
      target = target.cuda()
      logits = model(input)[0]
      pred = logits.argmax(dim=-1)
      train_prediction.extend(pred.tolist())


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
  

if __name__ == '__main__':
  main()

