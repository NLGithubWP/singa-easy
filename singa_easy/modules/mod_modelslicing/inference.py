import os
import argparse
import time
import shutil
from collections import OrderedDict
import importlib

import torch
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from data_loader import data_loader
from utils.utilities import logger, AverageMeter, accuracy, timeSince, accuracy_float
from utils.lr_scheduler import GradualWarmupScheduler
from models import upgrade_dynamic_layers, create_sr_scheduler
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(
    description='CIFAR-10, CIFAR-100 and ImageNet-1k Model Slicing Training')
parser.add_argument(
    '--exp_name',
    default='',
    type=str,
    help='optional exp name used to store log and checkpoint (default: none)')
parser.add_argument('--net_type',
                    default='resnet',
                    type=str,
                    help='network type: vgg, resnet, and so on')
parser.add_argument(
    '--groups',
    default=8,
    type=int,
    help='group num for Group Normalization (default 8, set to 0 for MultiBN)')
parser.add_argument('--depth',
                    default=50,
                    type=int,
                    help='depth of the network')
parser.add_argument('--arg1',
                    default=1.0,
                    type=float,
                    metavar='M',
                    help='additional model arg, k for ResNet')

parser.add_argument('--sr_list',
                    nargs='+',
                    help='the slice rate list in descending order',
                    required=True)
parser.add_argument('--sr_train_prob',
                    nargs='+',
                    help='the prob of picking subnet corresponding to sr_list')
parser.add_argument(
    '--sr_scheduler_type',
    default='random',
    type=str,
    help='slice rate scheduler, support random, random[_min][_max], round_robin'
)
parser.add_argument(
    '--sr_rand_num',
    default=1,
    type=int,
    metavar='N',
    help='the number of random sampled slice rate except min/max (default: 1)')

parser.add_argument('--epoch',
                    default=300,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size',
                    '-b',
                    default=128,
                    type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay',
                    '--wd',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--lr',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate')
parser.add_argument('--cosine',
                    dest='cosine',
                    action='store_true',
                    help='cosine LR scheduler')
parser.add_argument('--warmup',
                    dest='warmup',
                    action='store_true',
                    help='gradual warmup LR scheduler')
parser.add_argument('--lr_multiplier',
                    default=10.,
                    type=float,
                    metavar='LR',
                    help='LR warm up multiplier')
parser.add_argument('--warmup_epoch',
                    default=5,
                    type=int,
                    metavar='N',
                    help='LR warm up epochs')

parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '--resume_best',
    dest='resume_best',
    action='store_true',
    help='whether to resume the best_checkpoint (default: False)')
parser.add_argument('--checkpoint_dir',
                    default='~/checkpoint/',
                    type=str,
                    metavar='PATH',
                    help='path to checkpoint')

parser.add_argument('--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--data_dir',
                    default='./data/',
                    type=str,
                    metavar='PATH',
                    help='path to dataset')
parser.add_argument('--log_dir',
                    default='./log/',
                    type=str,
                    metavar='PATH',
                    help='path to log')
parser.add_argument('--dataset',
                    dest='dataset',
                    default='cifar10',
                    type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument(
    '--no_augment',
    dest='augment',
    action='store_false',
    help='whether to use standard augmentation for the datasets (default: True)'
)

parser.add_argument('--log_freq',
                    default=10,
                    type=int,
                    metavar='N',
                    help='log frequency')

parser.set_defaults(cosine=False)
parser.set_defaults(warmup=False)
parser.set_defaults(resume_best=False)
parser.set_defaults(augment=True)

# initialize all global variables
args = parser.parse_args()
args.data_dir += args.dataset
args.sr_list = list(map(float, args.sr_list))
if args.sr_train_prob:
    args.sr_train_prob = list(map(float, args.sr_train_prob))
if not args.exp_name:
    args.exp_name = '{0}_{1}_{2}'.format(args.net_type, args.depth,
                                         args.dataset)
args.checkpoint_dir = '{0}{1}/'.format(os.path.expanduser(args.checkpoint_dir),
                                       args.exp_name)
args.log_path = '{0}{1}/log.txt'.format(args.log_dir, args.exp_name)
best_err1, best_err5 = 100., 100.

# create log dir
if not os.path.isdir('log'):
    os.mkdir('log')

if not os.path.isdir('log/{}'.format(args.exp_name)):
    os.mkdir('log/{}'.format(args.exp_name))

# load dataset
train_loader, val_loader, args.class_num = data_loader(args)


def main():
    global args, best_err1, best_err5
    print_logger = logger(args.log_path, True, True)
    print(vars(args))

    # create model and upgrade model to support model slicing
    model = create_model(args, print_logger)
    model = upgrade_dynamic_layers(model, args.groups, args.sr_list)
    model = torch.nn.DataParallel(model).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    if args.resume:
        checkpoint = load_checkpoint(print_logger)
        epoch, best_err1, best_err5, model_state, optimizer_state, scheduler_state = checkpoint.values()
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        print("==> finish loading checkpoint '{}' (epoch {})".format(args.resume, epoch))
    cudnn.benchmark = True
    # evaluate on all the sr_idxs, from the smallest subnet to the largest
    for sr_idx in reversed(range(len(args.sr_list))):
        args.sr_idx = sr_idx
        print("Begin", "---" * 20)
        print("Under slice rate ", args.sr_list[sr_idx], "---" * 5)
        model.module.update_sr_idx(sr_idx)
        correct_k = 0
        total_time = 0
        for i in range(256):
            for idx, (input, target) in enumerate(val_loader):
                if torch.cuda.is_available():
                    input = input.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                be = time.time()
                output = model(input)
                correct_k += accuracy_float(output, target, topk=(1, 1))
                total_time += time.time()-be
                break
        print("aaccuracy", correct_k/256)
        print("average_time", total_time/256)
        print("End", "---" * 20)


def create_model(args, print_logger):
    print("==> creating model '{}'".format(args.net_type))
    models = importlib.import_module('models')
    if args.dataset.startswith('cifar'):
        model = getattr(models, 'cifar_{0}'.format(args.net_type))(args)
    elif args.dataset == 'imagenet':
        model = getattr(models, 'imagenet_{0}'.format(args.net_type))(args)
    print('the number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    return model


def load_checkpoint(print_logger):
    print("==> loading checkpoint '{}'".format(args.resume))

    if os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume)
    elif args.resume == 'checkpoint':
        if args.resume_best:
            checkpoint = torch.load('{0}{1}'.format(args.checkpoint_dir,
                                                    'best_checkpoint.ckpt'))
        else:
            checkpoint = torch.load('{0}{1}'.format(args.checkpoint_dir,
                                                    'checkpoint.ckpt'))
    else:
        raise Exception("=> no checkpoint found at '{}'".format(args.resume))
    return checkpoint


if __name__ == '__main__':
    print(torch.cuda.is_available())
    main()




