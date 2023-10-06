from __future__ import print_function

import torch
# torch.set_printoptions(profile="full")
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import argparse
import logging
import random
import time
import os
import glob
import sys
import numpy as np
from thop import profile
from utils import *


class SmallModel(nn.Module):
    def __init__(self, hidden_node):
        super(SmallModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, hidden_node, kernel_size=5, padding=1),
            nn.BatchNorm2d(hidden_node),
            nn.ReLU(inplace=True)
        )
        self.layer_m1 = nn.MaxPool2d(kernel_size=5, stride=2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_node, 1, kernel_size=5, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer_m1(x)
        x = self.layer2(x)

        return x


def Unit(node):
    l = np.random.randint(2, 5, size=node)
    network = []
    for i in l:
        network.append(SmallModel(i))
    networks = nn.ModuleList(network)
    return networks



def Concat(node, x, networks):
    list = []
    input = torch.split(x, 1, dim=1)
    for i in range(node):
        list.append(F.relu(networks[i](input[i])))
    x = torch.cat((list), 1)
    return x


class cell(nn.Module):
    def __init__(self, input_nodes):
        super(cell, self).__init__()

        self.input_nodes = input_nodes

        self.layer0 = nn.Conv2d(self.input_nodes[0], self.input_nodes[1], kernel_size=5)
        self.networks1 = Unit(self.input_nodes[1])
        self.layer_m1 = nn.Sequential(nn.MaxPool2d(kernel_size=5, stride=2))

        self.fc_out = nn.Linear(32, self.input_nodes[2])

    def forward(self, x):
        x = self.layer0(x)
        x = Concat(self.input_nodes[1], x, self.networks1)
        x = self.layer_m1(x)

        x = x.view(x.size(0), -1)
        x = self.fc_out(x)

        return x


input_nodes = [1, 8, 10]

model = cell(input_nodes)

# print(model)

input = torch.randn(1, 1, 28, 28)
macs, params = profile(model, inputs=(input,))
print(macs * 2, params)
model = model.cuda()


def weights_init_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)
    # elif classname.find('Conv') != -1:
    #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # nn.init.constant_(m.bias, 0)


# model.apply(weights_init_uniform)


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

trainset = torchvision.datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
testset = torchvision.datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=True)
n_classes = 10
# trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True,transform=transform_train)
# testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, pin_memory=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, pin_memory=True, num_workers=0)

learn_rate = 0.01
epochs = 200
optimizer = optim.SGD([p for n, p in model.named_parameters() if p.requires_grad], lr=learn_rate, momentum=0.9,
                      weight_decay=0.0005)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
criterion = nn.CrossEntropyLoss()


def train(epoch):
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
        if batch_idx % 100 == 0:
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
            _, predicted = outputs.max(1)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            end_time = time.time()

        logging.info(
            'Train Epoch: %d   Model Time: %.03f s', epoch, end_time - start_time)
    return correct, test_loss / len(testloader)


save = 'test-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
create_exp_dir(save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if __name__ == '__main__':
    cudnn.benchmark = True
    torch.cuda.manual_seed(20)
    cudnn.enabled = True
    torch.manual_seed(20)
    max_correct = 0
    warmup = 10
    for epoch in range(epochs):
        scheduler.step()
        train(epoch)
        if epoch == warmup:
            optimizer = optim.SGD(
                [p for n, p in model.named_parameters() if p.requires_grad], lr=learn_rate, weight_decay=0.0005)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs - warmup)

        correct, loss = test(epoch)
        if correct > max_correct:
            max_correct = correct
            torch.save(model, os.path.join(save, 'weight.pth'))
        logging.info('Epoch %d correct: %d, Max correct %d, Loss %.06f', epoch, correct, max_correct, loss)