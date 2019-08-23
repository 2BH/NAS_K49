import os
import argparse
import utils.utils as utils
import datetime
import sys
import logging
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from models.cnn import torchModel
from utils.datasets import K49, KMNIST
from sklearn.metrics import balanced_accuracy_score
import tqdm

def main(model_config,
         args):
    """
    Training loop for configurableNet.
    :param model_config: network config (dict)
    :param data_dir: dataset path (str)
    :param num_epochs: (int)
    :param batch_size: (int)
    :param learning_rate: model optimizer learning rate (float)
    :param train_criterion: Which loss to use during training (torch.nn._Loss)
    :param model_optimizer: Which model optimizer to use during trainnig (torch.optim.Optimizer)
    :param data_augmentations: List of data augmentations to apply such as rescaling.
        (list[transformations], transforms.Composition[list[transformations]], None)
        If none only ToTensor is used
    :return:
    """
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    data_dir = args.data_dir
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate= args.learning_rate
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model_optimizer = opti_dict[args.optimizer]

    save_model_str=args.model_path


    # Device configuration
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)

    # create data augmentation
    train_transform, valid_transform = utils.data_transforms_Kuzushiji(args)
    data_augmentations = transforms.ToTensor()
      # Dataset
    if args.set == "KMNIST":
        train_data = KMNIST(args.data_dir, True, train_transform)
        test_data = KMNIST(args.data_dir, False, valid_transform)
        #train_data = KMNIST(args.data_dir, True, data_augmentations)
        #test_data = KMNIST(args.data_dir, False, data_augmentations)
    elif args.set == "K49":
        #train_data = K49(args.data_dir, True, data_augmentations)
        #test_data = K49(args.data_dir, False, data_augmentations)
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
        print("Without weighted sample")
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            pin_memory=True, num_workers=2)

        valid_queue = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False,
            pin_memory=True, num_workers=2)

    # instantiate model
    model = torchModel(model_config,
                       input_shape=(
                           train_data.channels,
                           train_data.img_rows,
                           train_data.img_cols
                       ),
                       num_classes=train_data.n_classes
            ).cuda()
    total_model_params = np.sum(p.numel() for p in model.parameters())
    # instantiate optimizer
    optimizer = model_optimizer(model.parameters(),
                                lr=learning_rate,
                                #weight_decay=args.weight_decay
                                )
    # use cosine LR scheduler
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #    optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    logging.info('Generated Network:')

    best_acc = 0.0
    # Train the model
    for epoch in range(num_epochs):
        logging.info('#' * 50)
        logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

        train_score, train_loss = train(train_queue, model, criterion, optimizer)
        #scheduler.step()
        logging.info('(Unbalanced) Train accuracy %f', train_score)
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        is_best = False
        if valid_acc > best_acc:
            best_acc = valid_acc
            is_best = True
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc_top1': train_score,
            'optimizer' : optimizer.state_dict(),
            }, is_best, log_path)
    test_prediction = []

    for step, (input, target) in tqdm.tqdm(enumerate(valid_queue), disable=True):
        input = input.cuda()
        target = target.cuda()
        logits = model(input)
        pred = logits.argmax(dim=-1)
        test_prediction.extend(pred.tolist())
    test_labels = test_data.labels
    test_acc_bal = balanced_accuracy_score(test_labels, np.array(test_prediction))

    logging.info("(balanced) test acc: %f", test_acc_bal)

def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in tqdm.tqdm(enumerate(train_queue), disable=True):
        input = input.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        (prec1,) = utils.accuracy(logits, target, topk=(1,))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  model.eval()
  with torch.no_grad():
    for step, (input, target) in tqdm.tqdm(enumerate(valid_queue), disable=True):
      input = input.cuda()
      target = target.cuda()

      logits = model(input)
      loss = criterion(logits, target)

      (prec1,) = utils.accuracy(logits, target, topk=(1,))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f', step, objs.avg, top1.avg)

  return top1.avg, objs.avg



loss_dict = {'cross_entropy': torch.nn.CrossEntropyLoss,
                'mse': torch.nn.MSELoss}
opti_dict = {'adam': torch.optim.Adam,
                'adad': torch.optim.Adadelta,
                'sgd': torch.optim.SGD,
                'adamW': torch.optim.AdamW}

parser = argparse.ArgumentParser("kuzushiji")
parser.add_argument('--data_dir', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='Use weighted sampling')
parser.add_argument('--set', type=str, default='KMNIST', help='The dataset to be trained')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=20, help='num of training epochs')
parser.add_argument('--optimizer', type=str, default='adam', help='Type of optimizer to use')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=6, help='cutout length')
parser.add_argument('--log_dir', type=str, default='./log', help='logging file location')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=np.inf, help='gradient clipping')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')

args = parser.parse_args()

# architecture parametrization
architecture = {
        'n_layers': 1,
    }


# logging 
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.data_dir, exist_ok=True)

current_date = datetime.datetime.now()
timestamp = current_date.isoformat()

data_dir = args.data_dir
log_path = args.log_dir+'/exp_{}'.format(timestamp)
os.makedirs(log_path, exist_ok=True)
log_name = "log_test_" + args.set + ".txt"
log_dir = os.path.join(log_path, 'log_baseline_test_aux.txt')

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

main(architecture, args)
