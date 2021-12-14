'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import models

import numpy
import random
from gradinit_utils import gradinit, metainit
from utils import Cutout, mixup_criterion, mixup_data


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet110', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet110)')
parser.add_argument('--resume', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--seed', default=0, type=int,
                    help='rng seed')
parser.add_argument('--alpha', default=1., type=float,
                    help='interpolation strength (uniform=1., ERM=0.)')
parser.add_argument('--wd', default=1e-4, type=float,
                    help='weight decay (default=1e-4)')
parser.add_argument('--batchsize', default=128, type=int,
                    help='batch size per GPU (default=128)')
parser.add_argument('--n_epoch', default=200, type=int,
                    help='total number of epochs')
parser.add_argument('--base_lr', default=0.1, type=float,
                    help='base learning rate (default=0.1)')
parser.add_argument('--train-clip', default=-1, type=float,
                    help='Clip the gradient norm during training.')
parser.add_argument('--expname', default="default", type=str)
parser.add_argument('--no_bn', default=False, action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--cutout', default=False, action='store_true')
parser.add_argument('--train-loss', default='ce', type=str, choices=['ce', 'mixup'])

parser.add_argument('--metainit', default=False, action='store_true',
                    help='Whether to use MetaInit.')
parser.add_argument('--gradinit', default=False, action='store_true',
                    help='Whether to use GradInit.')
parser.add_argument('--gradinit-lr', default=1e-3, type=float,
                    help='The learning rate of GradInit.')
parser.add_argument('--gradinit-iters', default=390, type=int,
                    help='Total number of iterations for GradInit.')
parser.add_argument('--gradinit-alg', default='sgd', type=str,
                    help='The target optimization algorithm, deciding the direction of the first gradient step.')
parser.add_argument('--gradinit-eta', default=0.1, type=float,
                    help='The eta in GradInit.')
parser.add_argument('--gradinit-min-scale', default=0.01, type=float,
                    help='The lower bound of the scaling factors.')
parser.add_argument('--gradinit-grad-clip', default=1, type=float,
                    help='Gradient clipping (per dimension) for GradInit.')
parser.add_argument('--gradinit-gamma', default=float('inf'), type=float,
                    help='The gradient norm constraint.')
parser.add_argument('--gradinit-normalize-grad', default=False, action='store_true',
                    help='Whether to normalize the gradient for the algorithm A.')
parser.add_argument('--gradinit-resume', default='', type=str,
                    help='Path to the gradinit or metainit initializations.')
parser.add_argument('--gradinit-bsize', default=-1, type=int,
                    help='Batch size for GradInit. Set to -1 to use the same batch size as training.')
parser.add_argument('--batch-no-overlap', default=False, action='store_true',
                    help=r'Whether to make \tilde{S} and S different.')
args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
numpy.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = int(args.batchsize)
base_learning_rate = args.base_lr * args.batchsize / 128.
if use_cuda:
    # data parallel
    n_gpu = torch.cuda.device_count()

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.cutout:
    transform_train.transforms.append(Cutout(n_holes=1, length=16))

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset.lower() == 'cifar10':
    dset_class = torchvision.datasets.CIFAR10
    num_class = 10
elif args.dataset.lower() == 'cifar100':
    dset_class = torchvision.datasets.CIFAR100
    num_class = 100

trainset = dset_class(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = dset_class(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("=> creating model '{}'".format(args.arch))
net = models.__dict__[args.arch](use_bn=not args.no_bn, num_classes=num_class)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print('Using', torch.cuda.device_count(), 'GPUs.')
    cudnn.benchmark = True
    print('Using CUDA..')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    # checkpoint_file = './checkpoint/ckpt.t7.' + args.sess + '_' + str(args.seed)
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'].state_dict())
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])

result_folder = './results/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

if 'nobn' in args.arch or 'fixup' in args.arch or args.no_bn and 'resnet' in args.arch:
    parameters_bias = [p[1] for p in net.named_parameters() if 'bias' in p[0]]
    parameters_scale = [p[1] for p in net.named_parameters() if 'scale' in p[0]]
    parameters_others = [p[1] for p in net.named_parameters() if not ('bias' in p[0] or 'scale' in p[0] or 'autoinit' in p[0])]
    optimizer = optim.SGD(
            [{'params': parameters_bias, 'lr': args.base_lr/10.},
            {'params': parameters_scale, 'lr': args.base_lr/10.},
            {'params': parameters_others}],
            lr=base_learning_rate,
            momentum=0.9,
            weight_decay=args.wd)
else:
    bn_names = ['norm', 'bn']
    bn_params = []
    other_params = []
    bn_param_names = []
    for n, p in net.named_parameters():
        if any([k in n for k in bn_names]):
            bn_params.append(p)
            bn_param_names.append(n)
        else:
            other_params.append(p)
    optimizer = optim.SGD(
        [{'params': bn_params, 'weight_decay': 0},
         {'params': other_params, 'weight_decay': args.wd}],
        lr=base_learning_rate,
        momentum=0.9)

total_params = sum([p.numel() for p in net.parameters()])
print(">>>>>>>>>>>>>>> Total number of parameters: {}".format(total_params))

if args.gradinit:
    gradinit_bsize = int(args.batchsize / 2) if args.gradinit_bsize < 0 else int(args.gradinit_bsize / 2)
    gradinit_trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=gradinit_bsize,
        shuffle=True)

    gradinit(net, gradinit_trainloader, args)

if args.metainit:
    if args.arch == 'gradinit_resnet110':
        gradinit_trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=int(args.batchsize / 2),
            shuffle=True)
    elif args.arch == 'gradinit_densenet100':
        gradinit_trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=int(args.batchsize / 3),
            shuffle=True)
    else:
        gradinit_trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batchsize,
            shuffle=True)
    metainit(net, gradinit_trainloader, args)

cel = nn.CrossEntropyLoss()
criterion = lambda pred, target, lam: (
            -F.log_softmax(pred, dim=1) * torch.zeros(pred.size()).cuda().scatter_(1, target.data.view(-1, 1),
                                                                                   lam.view(-1, 1))).sum(dim=1).mean()


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    total_gnorm = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # generate mixed inputs, two one-hot label vectors and mixing coefficient
        optimizer.zero_grad()

        if args.train_loss == 'mixup':
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.alpha, use_cuda)
            outputs = net(inputs)

            loss_func = mixup_criterion(targets_a, targets_b, lam)
            loss = loss_func(criterion, outputs)
        else:
            outputs = net(inputs)
            loss = cel(outputs, targets)

        loss.backward()

        if args.train_clip > 0:
            gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), args.train_clip)
        else:
            gnorm = -1
        total_gnorm += gnorm

        optimizer.step()
        sgdr.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).cpu().sum()
        acc = 100.*float(correct)/float(total)

        if batch_idx % 50 == 0 or batch_idx == len(trainloader) - 1:
            wnorms = [w.norm().item() for n, w in net.named_parameters() if 'weight' in n]
            print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | WNorm: %.3e (min: %.3e, max: %.3e) | GNorm: %.3e (%.3e)'
                % (train_loss/(batch_idx+1), acc, correct, total, sum(wnorms), min(wnorms), max(wnorms), gnorm, total_gnorm / (batch_idx+1)))

    return train_loss/batch_idx, acc


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if batch_idx % 50 == 0 or batch_idx == len(testloader) - 1:
                print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total))

        # Save checkpoint.
        acc = 100.*float(correct)/float(total)
        if acc > best_acc:
            best_acc = acc
            checkpoint(acc, epoch)

    return test_loss/batch_idx, acc


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'optimizer': optimizer.state_dict()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + args.expname + '.ckpt')


sgdr = CosineAnnealingLR(optimizer, args.n_epoch * len(trainloader), eta_min=0, last_epoch=-1)

chk_path = os.path.join('chks', args.expname + "_latest.pth")
for epoch in range(start_epoch, args.n_epoch):
    lr = 0.

    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    print("Epoch {}, lr {}".format(epoch, lr))

    torch.save({'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, 'test_acc': test_acc},
               chk_path)
