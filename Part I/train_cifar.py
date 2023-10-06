import time
import os
import argparse
import logging
import glob
import sys
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from thop import profile
from dataload import data_load
from bunitnet import *

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
          dst_file = os.path.join(path, 'scripts', os.path.basename(script))
          shutil.copyfile(script, dst_file)

def weights_init_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 0.01)

def train_one_epoch(epoch, args):
    global model
    logging.info('\nepoch: %d, Learning rate: %f', epoch, scheduler.get_last_lr()[0])
    model.train()
    train_loss = 0

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
            model = model.cuda()
        data_time = time.time()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        model_time = time.time()

        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: %d  Process: %d  Total: %d  Loss: %.06f  Data Time: %.03fs  Model Time: %.03fs',
                         epoch, batch_idx * len(inputs), len(trainloader.dataset), loss.item(), data_time-end, model_time-data_time)
        end = time.time()

def test_one_epoch(epoch):
    global model
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
                model = model.cuda()
            start_time = time.time()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            end_time = time.time()

        logging.info('Train Epoch: %d Model Time: %.03fs', epoch, end_time-start_time)
    return correct, test_loss / len(testloader)

def set_env(args):
    cudnn.benchmark = True
    torch.cuda.manual_seed(args.seed)
    cudnn.enabled = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.save = '{}'.format(time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    print(args)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("args = %s", args)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in args.gpu])
    return args, logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--dataname', default='cifar10', type=str)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--seed', type=int, default=30)
    parser.add_argument('--gpu', default=[1], type=str)
    args = parser.parse_args()

    args, logging = set_env(args) 
    print('==> Preparing dataset...')
    trainloader, testloader, class_num = data_load(args)

    print('==> Building model...')
    unit_num = [20, 20]
    model = BunitNet(3, unit_num, class_num) # 1 for mnist
    # print(model)
    model.apply(weights_init_uniform)

    print('==> Starting training...')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([p for n, p in model.named_parameters() if p.requires_grad], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.Adam([p for n, p in model.named_parameters() if p.requires_grad], lr=args.lr,weight_decay=args.weight_decay,betas=(0.9, 0.99), eps=1e-02)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    max_correct = 0
    for epoch in range(args.epochs):
        if epoch == args.warmup:
            optimizer = optim.SGD([p for n, p in model.named_parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
            # optimizer = optim.Adam([p for n, p in model.named_parameters() if p.requires_grad], lr=args.lr,weight_decay=args.weight_decay,betas=(0.9, 0.99), eps=1e-02)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - args.warmup)

        train_one_epoch(epoch, args)
        scheduler.step()

        correct, loss = test_one_epoch(epoch)
        # print('epoch: {} acc: {}'.format(epoch, correct))
        if correct > max_correct:
            max_correct = correct
            torch.save(model, os.path.join(args.save, 'weights.pth'))
        logging.info('epoch %d correct: %d, Max correct %d, Loss %.06f', epoch, correct, max_correct, loss)




