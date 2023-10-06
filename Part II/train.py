from __future__ import print_function
import pickle
import torch
# torch.set_printoptions(profile="full")
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import random
import time
import os
import argparse
import logging
import glob
import sys
import matplotlib.pyplot as plt

from thop import profile
from utils import *

from bunit_resnet import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--gamma', default=0.1, type=float)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1024)

parser.add_argument('--gpu', default=[0], type=list)

parser.add_argument('--log_interval', default=100, type=int)
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--kernel_size', type=int, default=3, help='3 for resnet like model, 1 for mobilelike model')
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--seed', type=int, default=4)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--warmup', type=int, default=10)


args = parser.parse_args()
args.save = 'test-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
print(args)

create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in args.gpu])

print('==> Preparing data..')


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True,transform=transform_train)  # download的False改成True
testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
n_classes = 100
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True,num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True,num_workers=0)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# Model
print('==> Building model..')


model = resnet18()


input = torch.randn(1, 3, 32,32)
macs, params = profile(model, inputs=(input, ))
print(macs*2, params)

logging.info("args = %s", args)


criterion = nn.CrossEntropyLoss()


def weights_init_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # nn.init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
            nn.init.normal_(m.weight, 0, 0.01)
            # nn.init.constant_(m.bias, 0)

model.apply(weights_init_uniform)
model = model.cuda()


optimizer = optim.SGD([p for n, p in model.named_parameters() if p.requires_grad], lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)


def train(epoch, args):
    logging.info('\nEpoch: %d, Learning rate: %f', epoch, scheduler.get_lr()[0])
    model.train()
    train_loss = 0

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        data_time = time.time()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        train_loss += loss.item()


        model_time = time.time()
        if batch_idx % args.log_interval == 0:
            logging.info(
                'Train Epoch: %d Process: %d Total: %d    Loss: %.06f    Data Time: %.03f s    Model Time: %.03f s',
                epoch, batch_idx * len(inputs), len(trainloader.dataset), loss.item(), data_time - end,
                model_time - data_time)
        end = time.time()
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()

            start_time = time.time()
            outputs = model(inputs)


            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            end_time = time.time()

        logging.info(
            'Train Epoch: %d   Model Time: %.03f s ',epoch, end_time - start_time)

    return correct, test_loss / len(testloader)

if __name__ == '__main__':
    cudnn.benchmark = True
    torch.cuda.manual_seed(args.seed)
    cudnn.enabled = True
    torch.manual_seed(args.seed)
    max_correct = 0



    for epoch in range(args.epochs):

        if epoch == args.warmup:

            optimizer = optim.SGD(
                [p for n, p in model.named_parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - args.warmup)
        scheduler.step()
        train(epoch, args)
        correct, loss = test(epoch)

        if correct > max_correct:
            max_correct = correct
            torch.save(model, os.path.join(args.save, 'weights.pth'))
        logging.info('Epoch %d correct: %d, Max correct %d, Loss %.06f', epoch, correct, max_correct, loss)

