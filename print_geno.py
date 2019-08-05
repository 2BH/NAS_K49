import os
import torch
import argparse
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.model_search import Network
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser("kmnist")
parser.add_argument('--data_dir', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='KMNIST', help='The dataset to be trained')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--input_channels', type=int, default=1, help='num of input channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--log_dir', type=str, default='./log', help='logging file location')
parser.add_argument('--seed', type=int, default=2, help='random seed')
args = parser.parse_args()


os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.data_dir, exist_ok=True)

timestamp = "2019-08-03T23:15:04.040061"

data_dir = args.data_dir
log_path = args.log_dir+'/exp_{}'.format(timestamp)
os.makedirs(log_path, exist_ok=True)


if args.set == 'KMNIST':
  num_classes = 10
elif args.set == 'K49':
  num_classes = 49
else:
  raise ValueError("Invalid dataset name %s" % args.set)

def main():
  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  print("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  model = Network(args.init_channels, args.input_channels, num_classes, args.layers, criterion)

  model = model.cuda()
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  model.load_state_dict(torch.load(log_path + '/weights.pt'))

  print(model.genotype())

if __name__ == "__main__":
  main()
