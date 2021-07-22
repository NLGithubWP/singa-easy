import os
import argparse
import torch.nn as nn

import importlib

import torch
import torch.backends.cudnn as cudnn

from data_loader import data_loader
from utils.utilities import logger, AverageMeter, accuracy, timeSince, accuracy_float
from models import upgrade_dynamic_layers


import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(
    description='CIFAR-10, CIFAR-100 and ImageNet-1k Model Slicing Training')

parser.add_argument(
    '--predict_batch_nums',
    default=1,
    type=int,
    help='number of processing a batch of images.')


parser.add_argument(
    '--predicted_save_file',
    default="a",
    type=str,
    help='save file for predicted images')


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

def create_model(args, print_logger):
    print_logger.info("==> creating model '{}'".format(args.net_type))
    if args.dataset.startswith('cifar'):
        models = importlib.import_module('models')
        model = getattr(models, 'cifar_{0}'.format(args.net_type))(args)
        print_logger.info('the number of model parameters: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])))
        return model

    elif args.dataset == 'imagenet':
        models = importlib.import_module('models')
        model = getattr(models, 'imagenet_{0}'.format(args.net_type))(args)
        print_logger.info('the number of model parameters: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])))
        return model

    elif args.dataset == 'xray':
        from torchvision import models
        resnet50 = models.resnet50(pretrained=True)

        # 不对上面的层进行训练 c
        for param in resnet50.parameters():
            param.requires_grad = False

        fc_inputs = resnet50.fc.in_features
        # 修改最后一层
        resnet50.fc = torch.nn.Sequential(
            torch.nn.Linear(fc_inputs, 2),
            torch.nn.LogSoftmax(dim=1)
        )
        return resnet50

    elif args.dataset == 'food':
        from torchvision import models
        resnet50 = models.resnet50(pretrained=True)

        # 不对上面的层进行训练 c
        for param in resnet50.parameters():
            param.requires_grad = False
        fc_inputs = resnet50.fc.in_features
        resnet50.fc = nn.Sequential(
            nn.Linear(fc_inputs, 58),
            nn.LogSoftmax(dim=1)
        )
        return resnet50

    else:
        raise


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
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    # warmup
    for sr_idx in reversed(range(len(args.sr_list))):
        args.sr_idx = sr_idx
        model.module.update_sr_idx(sr_idx)
        for idx, (input, target) in enumerate(val_loader):
            print("GPU-WARM-UP batchid", idx, "Under slice rate ", args.sr_list[sr_idx], "---" * 5)
            if torch.cuda.is_available():
                input = input.cuda(non_blocking=True)
                target.cuda(non_blocking=True)
            model(input)
            torch.cuda.synchronize()
            break

    print("GPU-WARM-UP done")

    test_scheduler_batch_examples(starter, ender, model)


def test_scheduler_batch_examples(starter, ender, model):
    scheduler_map = \
        {32: [32, 0, 0, 0], 160: [160, 0, 0, 0], 320: [320, 0, 0, 0], 640: [640, 0, 0, 0], 960: [960, 0, 0, 0],
         1280: [1280, 0, 0, 0], 1600: [1600, 0, 0, 0], 1920: [1920, 0, 0, 0], 2240: [2240, 0, 0, 0], 2560: [2560, 0, 0, 0],
         3200: [3200, 0, 0, 0], 3840: [3840, 0, 0, 0], 4480: [4480, 0, 0, 0], 4800: [4800, 0, 0, 0], 5120: [5120, 0, 0, 0],
         5440: [5440, 0, 0, 0], 5760: [5586, 0, 174, 0], 6400: [4937, 0, 1463, 0], 8000: [3314, 0, 4686, 0],
         9600: [1691, 0, 7909, 0], 11200: [68, 1, 11131, 0], 12800: [0, 0, 7854, 4946], 17600: [0, 0, 1, 16325],
         19200: [0, 0, 1, 16325], 20800: [0, 0, 1, 16325], 22400: [0, 0, 1, 16325], 24000: [0, 0, 1, 16325],
         25600: [0, 0, 1, 16325], 32000: [0, 0, 1, 16325]}

    sliceRateMapper = {0.25: 3,
                       0.5: 2,
                       0.75: 1,
                       1.0: 0}

    result = []
    fo = open(args.predicted_save_file + ".txt", "a+")
    max_images_num = args.predict_batch_nums * args.batch_size
    fo.write("When num_img/batch_size=" + str(args.predict_batch_nums) + "\n")
    fo.write("And max_images_num=" + str(max_images_num) + "\n")

    if max_images_num in scheduler_map:
        n1, n2, n3, n4 = scheduler_map[max_images_num]
        fo.write("Schedule result=[" +
                 str(n1) + " " +
                 str(n2) + " " +
                 str(n3) + " " +
                 str(n4) + " " + "]"
                 "\n")

    else:
        fo.write("     scheduler not support the current workload\n")
        fo.close()
        return

    correct_k = 0
    total_time = 0
    num_img = 0
    num_batch = 0

    if n1 != 0:
        print("Switch to model with slice-rate=", sliceRateMapper[1.0])
        model.module.update_sr_idx(sliceRateMapper[1.0])
        correct_k_tmp, num_img_tmp, total_time_tmp, num_batch_tmp = test_1_batch_examples(starter, ender, model, n1)

        correct_k += correct_k_tmp
        total_time += total_time_tmp
        num_img += num_img_tmp

    if n2 != 0:
        print("Switch to model with slice-rate=", sliceRateMapper[0.75])
        model.module.update_sr_idx(sliceRateMapper[0.75])
        correct_k_tmp, num_img_tmp, total_time_tmp, num_batch_tmp = test_1_batch_examples(starter, ender, model, n2)

        correct_k += correct_k_tmp
        total_time += total_time_tmp
        num_img += num_img_tmp

    if n3 != 0:
        print("Switch to model with slice-rate=", sliceRateMapper[0.5])
        model.module.update_sr_idx(sliceRateMapper[0.5])
        correct_k_tmp, num_img_tmp, total_time_tmp, num_batch_tmp = test_1_batch_examples(starter, ender, model, n3)

        correct_k += correct_k_tmp
        total_time += total_time_tmp
        num_img += num_img_tmp

    if n4 != 0:
        print("Switch to model with slice-rate=", sliceRateMapper[0.25])
        model.module.update_sr_idx(sliceRateMapper[0.25])
        correct_k_tmp, num_img_tmp, total_time_tmp, num_batch_tmp = test_1_batch_examples(starter, ender, model, n4)

        correct_k += correct_k_tmp
        total_time += total_time_tmp
        num_img += num_img_tmp

    num_batch = num_img/args.batch_size
    result.append([correct_k, num_img, total_time, "scheduler", num_batch])

    for ele in result:
        correct_k, num_img, total_time, sr_idx, num_batch = ele[0], ele[1], ele[2], ele[3], ele[4]
        print("sr_idx=", sr_idx, " correct_k", correct_k)
        print("sr_idx=", sr_idx, " num_img", num_img)
        print("sr_idx=", sr_idx, " accuracy", correct_k / num_img)
        print("sr_idx=", sr_idx, " average_time", total_time / num_img)
        print("sr_idx=", sr_idx, " throughput", num_img / total_time)
        print("sr_idx=", sr_idx, " End", "---" * 20)

        fo.write("     sr_idx=" + str(sr_idx) + " num_img=" + str(num_img) + "\n")
        fo.write("     sr_idx=" + str(sr_idx) + " correct_k=" + str(correct_k) + "\n")
        fo.write("     sr_idx=" + str(sr_idx) + " accuracy=" + str(correct_k / num_img) + "\n")
        fo.write("     sr_idx=" + str(sr_idx) + " average_time=" + str(total_time / num_img) + "\n")
        fo.write("     sr_idx=" + str(sr_idx) + " total_time=" + str(total_time) + "\n")
        fo.write("     sr_idx=" + str(sr_idx) + " throughput=" + str(num_img / total_time) + "\n")

        fo.write("     sr_idx=" + str(sr_idx) + " average_batch_time=" + str(total_time / num_batch) + "\n")
        fo.write("     sr_idx=" + str(sr_idx) + " throughput_batch=" + str(num_batch / total_time) + "\n")

        fo.write("\n")
    fo.close()


def log_test_1_batch_examples(model, starter, ender):
    result = []
    fo = open(args.predicted_save_file + ".txt", "a+")

    fo.write("When num_img/batch_size=" + str(args.predict_batch_nums) + "\n")

    for sr_idx in reversed(range(len(args.sr_list))):
        args.sr_idx = sr_idx
        print("Begin", "---" * 20)
        print("Under slice rate ", args.sr_list[sr_idx], "---" * 5)
        model.module.update_sr_idx(sr_idx)
        correct_k, num_img, total_time, num_batch = test_1_batch_examples(starter, ender, model, args.predict_batch_nums*args.batch_size)
        result.append([correct_k, num_img, total_time, args.sr_list[sr_idx], num_batch])

    for ele in result:
        correct_k, num_img, total_time, sr_idx, num_batch = ele[0], ele[1], ele[2], ele[3], ele[4]
        print("sr_idx=", sr_idx, " correct_k", correct_k)
        print("sr_idx=", sr_idx, " num_img", num_img)
        print("sr_idx=", sr_idx, " accuracy", correct_k / num_img)
        print("sr_idx=", sr_idx, " average_time", total_time / num_img)
        print("sr_idx=", sr_idx, " throughput", num_img / total_time)
        print("sr_idx=", sr_idx, " End", "---" * 20)

        fo.write("     sr_idx=" + str(sr_idx) + " num_img=" + str(num_img) + "\n")
        fo.write("     sr_idx=" + str(sr_idx) + " correct_k=" + str(correct_k) + "\n")
        fo.write("     sr_idx=" + str(sr_idx) + " accuracy=" + str(correct_k / num_img) + "\n")
        fo.write("     sr_idx=" + str(sr_idx) + " average_time=" + str(total_time / num_img) + "\n")
        fo.write("     sr_idx=" + str(sr_idx) + " total_time=" + str(total_time) + "\n")
        fo.write("     sr_idx=" + str(sr_idx) + " throughput=" + str(num_img / total_time) + "\n")

        fo.write("     sr_idx=" + str(sr_idx) + " average_batch_time=" + str(total_time / num_batch) + "\n")
        fo.write("     sr_idx=" + str(sr_idx) + " throughput_batch=" + str(num_batch / total_time) + "\n")

        fo.write("\n")
    fo.close()

def test_1_batch_examples(starter, ender, model, maximg):
    correct_k = 0
    total_time = 0
    num_img = 0
    num_batch = 0
    is_stop = False
    for i in range(99999999):
        for idx, (input, target) in enumerate(val_loader):
            print("Size is ", target.size(), int(target.size()[0]))
            if int(target.size()[0]) != 32:
                continue
            if torch.cuda.is_available():
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            starter.record()
            output = model(input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            total_time += curr_time
            num_img += args.batch_size
            num_batch += 1
            print("image number", num_img, " idx=", idx, "num_batch=", num_batch)
            correct_k += accuracy_float(output, target, topk=(1, 1))
            if num_img >= maximg:
                is_stop = True
                break

        if is_stop == True:
            break
    return correct_k, num_img, total_time, num_batch


def test_influence(starter, ender, model):
    correct_k = 0
    total_time = 0
    num_img = 0
    num_batch = 0
    is_stop = False
    for i in range(100000):
        for idx, (input, target) in enumerate(val_loader):
            if idx==4:
                if torch.cuda.is_available():
                    input = input.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                starter.record()
                output = model(input)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                total_time += curr_time
                num_img += args.batch_size
                num_batch += 1
                print("image number", num_img, " idx=", idx, "num_batch=", num_batch)
                correct_k += accuracy_float(output, target, topk=(1, 1))
                if num_img >= args.predict_batch_nums*args.batch_size:
                    is_stop = True
                    break

        if is_stop == True:
            break
    return correct_k, num_img, total_time, num_batch


if __name__ == '__main__':
    print(torch.cuda.is_available())
    main()




