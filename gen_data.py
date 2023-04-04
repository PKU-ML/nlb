import argparse
import logging
import sys
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.utils.tensorboard.writer import SummaryWriter

import os

from wideresnet import WideResNet
from preactresnet import PreActResNet18

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='PreActResNet18')
parser.add_argument('--l2', default=0, type=float)
parser.add_argument('--l1', default=0, type=float)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--data-dir', default='/data/yfwang/cifar-data', type=str)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--lr-schedule', default='piecewise', 
# choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine', 'constant', 'cyclic']
)
parser.add_argument('--lr-max', default=0.1, type=float)
parser.add_argument('--lr-one-drop', default=0.01, type=float)
parser.add_argument('--lr-drop-epoch', default=100, type=int)
parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'free', 'none', 'random', 'fgsm_fix'])
parser.add_argument('--eval-attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'free', 'none', 'fgsm_fix'])
parser.add_argument('--epsilon', default=8, type=int)
parser.add_argument('--attack-iters', default=10, type=int)
parser.add_argument('--restarts', default=1, type=int)
parser.add_argument('--pgd-alpha', default=2, type=float)
parser.add_argument('--fgsm-alpha', default=1.25, type=float)
parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])
parser.add_argument('--fname', default='cifar_model', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--half', action='store_true')
parser.add_argument('--width-factor', default=10, type=int)
parser.add_argument('--resume', default=0, type=int)
parser.add_argument('--cutout', action='store_true')
parser.add_argument('--cutout-len', type=int, default=14)
parser.add_argument('--mixup', action='store_true')
parser.add_argument('--mixup-alpha', type=float, default=1.4)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--val', action='store_true')
parser.add_argument('--chkpt-iters', default=10, type=int)

parser.add_argument('--num-classes', default=10, type=int)
parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--reg-type', default=None, type=str)
parser.add_argument('--temp', type=float, default=1.0)
parser.add_argument('--reg-schedule', default='constant', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine', 'constant', 'cyclic'])
parser.add_argument('--beta-factor', type=float, default=2.0)
parser.add_argument('--load-folder', default='cifar_model', type=str)
# parser.add_argument('--beta-factor', type=float, default=2.0)
parser.add_argument('--gamma', type=float, default=0.0)
parser.add_argument('--load-epoch', default=-1, type=int)
parser.add_argument('--fix-attack', action='store_true')
parser.add_argument('--further-attack', action='store_true')
parser.add_argument('--start-epoch', default=-1, type=int)
parser.add_argument('--sub-classes', nargs='+', type=int, default=None)
parser.add_argument('--cycle-epoch', default=100, type=int)
parser.add_argument('--rot', action='store_true')
parser.add_argument('--only-false', action='store_true', default=True)
parser.add_argument('--lr-factor', type=float, default=10.0)

args = parser.parse_args()

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

def normalize(X):
    # import pdb; pdb.set_trace()
    return (X - mu)/std

upper_limit, lower_limit = 1,0


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.to(device).float(), 'target': y.to(device).long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None, targets=None, gamma=0.0, rand_init=True):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if rand_init:
            if norm == "l_inf":
                delta.uniform_(-epsilon, epsilon)
            elif norm == "l_2":
                delta.normal_()
                d_flat = delta.view(delta.size(0),-1)
                n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
                r = torch.zeros_like(n).uniform_(0, 1)
                delta *= r/n*epsilon
            else:
                raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
            else:
                if targets is None:
                    loss = F.cross_entropy(output, y)
                    if gamma > 0.0:
                        robust_prob = F.softmax(output / args.temp, dim=1)
                        reg_loss = classwise_std(robust_prob, y, args.num_classes)
                        loss -= reg_loss * gamma
                else:
                    loss = - F.cross_entropy(output, targets)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
        else:
            output = model(normalize(X+delta))
            if targets is None:
                all_loss = F.cross_entropy(output, y, reduction='none')
            else:
                all_loss = - F.cross_entropy(output, targets, reduction='none')
                if gamma > 0.0:
                    robust_prob = F.softmax(output / args.temp, dim=1)
                    reg_loss = classwise_std(robust_prob, y, args.num_classes)
                    all_loss -= reg_loss * gamma
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def main():

    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    transforms = []
    if args.cutout:
        transforms.append(Cutout(args.cutout_len, args.cutout_len))
    if args.val:
        try:
            dataset = torch.load("cifar10_validation_split.pth")
        except:
            print("Couldn't find a dataset with a validation split, did you run "
                  "generate_validation.py?")
            return
        val_set = list(zip(transpose(dataset['val']['data']/255.), dataset['val']['labels']))
        val_batches = Batches(val_set, args.batch_size, shuffle=False, num_workers=2)
    else:
        dataset = cifar10(args.data_dir, num_classes=args.num_classes, sub_classes=args.sub_classes)
    # train_set = list(zip(transpose(pad(dataset['train']['data'], 4)/255.),
    train_set = list(zip(transpose(dataset['train']['data'])/255.,
        dataset['train']['labels']))
    train_set_x = Transform(train_set, transforms)
    train_batches = Batches(train_set_x, args.batch_size, shuffle=False, set_random_choices=True, num_workers=2)

    test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
    test_batches = Batches(test_set, args.batch_size, shuffle=False, num_workers=2)

    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

    if args.model == 'PreActResNet18':
        model = PreActResNet18(num_classes=args.num_classes)
    elif args.model == 'WideResNet':
        model = WideResNet(34, 10, widen_factor=args.width_factor, dropRate=0.0)
    else:
        raise ValueError("Unknown model")

    model = nn.DataParallel(model).cuda()
    model.eval()
    attack_model = model

    if args.load_epoch >= 0:
        # load ckpt
        load_epoch = args.load_epoch
        attack_model.load_state_dict(torch.load(os.path.join(args.load_folder, f'model_{load_epoch}.pth')))
        logger.info('Attacker load ckpt from at {}'.format(os.path.join(args.load_folder, f'model_{load_epoch}.pth')))

    new_data, new_y = [], []
    train_loss = 0
    train_acc = 0
    train_robust_loss = 0
    train_reg_loss = 0
    train_robust_acc = 0
    train_n = 0
    true_y, pred_y, pred_y_rob = [], [], []

    for i, batch in enumerate(train_batches):
        X, y = batch['input'], batch['target']
        # import pdb; pdb.set_trace()

        if args.attack == 'pgd' or args.attack == 'random':
            if args.attack != 'random':
                targets = None
            else:
                targets = torch.randint_like(y, args.num_classes)
            # Random initialization
            if args.mixup:
                delta = attack_pgd(attack_model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, mixup=True, y_a=y_a, y_b=y_b, lam=lam)
            else:
                delta = attack_pgd(attack_model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, targets=targets, gamma=args.gamma)
            if args.further_attack:
                delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, targets=targets, gamma=args.gamma)
            delta = delta.detach()
        elif args.attack == 'fgsm':
            delta = attack_pgd(attack_model, X, y, epsilon, args.fgsm_alpha*epsilon, 1, 1, args.norm)
        elif args.attack == 'fgsm_fix':
            delta = attack_pgd(attack_model, X, y, epsilon, args.fgsm_alpha*epsilon, 1, 1, args.norm, rand_init=False)
        # Standard training
        elif args.attack == 'none':
            delta = torch.zeros_like(X)
        X_adv = torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)
       
        output = model(normalize(X))
        robust_output = model(normalize(X_adv))
        pred_y = robust_output.max(1)[1]
        train_robust_acc += (robust_output.max(1)[1] == y).sum().item()
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)

        # import pdb; pdb.set_trace()

        # if args.only_false:
        # new_data.append(X_adv[pred_y!=y].data.cpu())
        # # new_y.append(pred_y[pred_y!=y].data.cpu())
        # new_y.append(pred_y[pred_y!=y].data.cpu())
        # else:
        new_data.append(X_adv.data.cpu())
        new_y.append(y.data.cpu())
        # break
    
    train_acc = train_acc / train_n
    train_robust_acc = train_robust_acc / train_n
    print(f'train_acc {train_acc} robust_acc {train_robust_acc}')

    new_data = (torch.cat(new_data) * 255.0).type(torch.uint8).permute(0,2,3,1).numpy()
    new_y = torch.cat(new_y).type(torch.long).tolist()

    # torch.save([new_data, new_y], os.path.join(args.load_folder,  f'non_robust_epoch{args.load_epoch}.pt'))
    torch.save([new_data, new_y], os.path.join(args.load_folder,  f'cifar10_adv_epoch{args.load_epoch}.pt'))

if __name__ == "__main__":
    main()
