# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os

import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torchvision.models import resnet18, resnet50
import torchvision

from solo.args.setup import parse_args_linear, parse_args_pretrain
from solo.methods.base import BaseMethod
from solo.utils.backbones import (
    swin_base,
    swin_large,
    swin_small,
    swin_tiny,
    vit_base,
    vit_large,
    vit_small,
    vit_tiny,
)

from solo.utils.classification_dataloader import prepare_data_no_aug
from poisoning_utils import *
from PIL import Image


def main(args):

    assert args.backbone in BaseMethod._SUPPORTED_BACKBONES
    backbone_model = {
        "resnet18": resnet18,
        "resnet50": resnet50,
        "vit_tiny": vit_tiny,
        "vit_small": vit_small,
        "vit_base": vit_base,
        "vit_large": vit_large,
        "swin_tiny": swin_tiny,
        "swin_small": swin_small,
        "swin_base": swin_base,
        "swin_large": swin_large,
    }[args.backbone]

    # initialize backbone
    kwargs = args.backbone_args
    cifar = kwargs.pop("cifar", False)
    # swin specific
    if "swin" in args.backbone and cifar:
        kwargs["window_size"] = 4

    backbone = backbone_model(**kwargs)
    if "resnet" in args.backbone:
        # remove fc layer
        if args.load_linear:
            backbone.fc = nn.Linear(backbone.inplanes, args.num_classes)
        else:
            backbone.fc = nn.Identity()
        if cifar:
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            backbone.maxpool = nn.Identity()

    assert (
        args.pretrained_feature_extractor.endswith(".ckpt")
        or args.pretrained_feature_extractor.endswith(".pth")
        or args.pretrained_feature_extractor.endswith(".pt")
    )
    ckpt_path = args.pretrained_feature_extractor

    state = torch.load(ckpt_path)["state_dict"]

    for k in list(state.keys()):
        if "encoder" in k:
            raise Exception(
                "You are using an older checkpoint."
                "Either use a new one, or convert it by replacing"
                "all 'encoder' occurances in state_dict with 'backbone'"
            )
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        if args.load_linear:
            if "classifier" in k:
                state[k.replace("classifier.", "fc.")] = state[k]
        del state[k]
    # prepare model
    backbone.load_state_dict(state, strict=False)
    backbone = backbone.cuda()
    backbone.eval()

    train_loader, val_loader, train_dataset, val_dataset = prepare_data_no_aug(
        args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    torch.manual_seed(42)
    torch.random.manual_seed(42)
    torch.cuda.manual_seed_all(42)


    if args.label_noise:
        y = torch.tensor(train_dataset.targets)
        if args.label_noise == 'sym':
            mask = torch.rand(*y.size())<args.noise_ratio
            rand_y = (torch.randint(1, 10, y.size()) + y) % 10
            y[mask] = rand_y[mask]
        elif args.label_noise == 'sym2':
            mask = torch.rand(*y.size())<args.noise_ratio
            rand_y = torch.randint(0, 10, y.size())
            y[mask] = rand_y[mask]
        elif args.label_noise == 'asym':
            source_class = [9, 2, 3, 5, 4]
            target_class = [1, 0, 5, 3, 7]
            old_y = y.clone()
            for s, t in zip(source_class, target_class):
                class_mask = old_y == s
                y.masked_fill_(torch.logical_and(class_mask, torch.rand(*y.size())<args.noise_ratio), t)
        else:
            raise ValueError('label noise not known')
        train_dataset.targets = y
    # linear_eval(backbone, train_loader, val_loader, feat_dim=512, batch_size=args.eval_batch_size, num_classes=args.num_classes, lr=args.linear_lr, pca_dim=args.pca_dim, eval_weight_decay=args.eval_weight_decay)
    knn(backbone, train_loader, val_loader, feat_dim=512, batch_size=args.eval_batch_size, num_classes=args.num_classes, lr=args.linear_lr, pca_dim=args.pca_dim, eval_weight_decay=args.eval_weight_decay, K=args.K)


def knn(model, train_loader, test_loader, feat_dim, batch_size, num_classes, lr, pca_dim=None, eval_weight_decay=0.0, K=5):
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO
        )

    X_train, y_train = inference(train_loader, model)
    X_test, y_test = inference(test_loader, model)
    test_set = torch.utils.data.TensorDataset(
        X_test, y_test
    )
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(val_loader):
            # targets = targets.cuda(non_blocking=True)
            # batch_size = inputs.size(0)
            # features = self.model(inputs.to(self.device))

            dist = torch.mm(features, X_train.T)
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = y_train.view(1, -1).expand(batch_size, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)

            total += targets.size(0)
            correct += retrieval.eq(targets.data).sum().item()
    top1 = correct / total

    print(f'[{K}] acc {top1}')

    return top1


def linear_eval(model, train_loader, test_loader, feat_dim, batch_size, num_classes, lr, pca_dim=None, eval_weight_decay=0.0):
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO
        )

    X_train, y_train = inference(train_loader, model)
    X_test, y_test = inference(test_loader, model)

    if pca_dim:
        torch.svd(X_train)

    train_set = torch.utils.data.TensorDataset(
        X_train, y_train
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )

    test_set = torch.utils.data.TensorDataset(
        X_test, y_test
    )
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )

    classifier = nn.Linear(feat_dim, num_classes).cuda()
    classifier.weight.data.normal_(mean=0.0, std=0.01)
    classifier.bias.data.zero_()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, weight_decay=eval_weight_decay, momentum=0.9)

    best_acc1 = 0.
    for epoch in range(500):
        logging.info("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        # adjust_learning_rate(optimizer, e  poch, lr=lr)

        train(classifier, train_loader, optimizer, criterion)

        with torch.no_grad():
            acc1, acc5 = validate(classifier, val_loader)
        
        best_acc1 = max(acc1, best_acc1)
        logging.info(f'epoch {epoch} acc {acc1} best_acc1 {best_acc1}')
    
    print('Final Best acc:', best_acc1.item())
    return best_acc1


def train(classifier, train_loader, optimizer, criterion):
    for i, (feat, target) in enumerate(train_loader):
        feat, target = feat.cuda(), target.cuda()

        # compute output
        output = classifier(feat.cuda())
        loss = criterion(output, target.cuda()) 
        # + torch.abs(classifier.weight).mean()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(classifier, val_loader):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    for i, (feat, target) in enumerate(val_loader):
        feat, target = feat.cuda(), target.cuda()

        # compute output
        output = classifier(feat)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], feat.size(0))
        top5.update(acc5[0], feat.size(0))

    return top1.avg, top5.avg

def inference(loader, model):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.cuda()
        y = y.cuda()

        # get encoding
        with torch.no_grad():
            h = model(x)

        feature_vector.append(h.data)
        labels_vector.append(y)

        # if step % 50 == 0:
        #     print(f"Step [{step}/{len(loader)}]\t Computing features...")


    feature_vector = torch.cat(feature_vector)
    labels_vector = torch.cat(labels_vector)
    # print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector

def create_feature_loaders(model, train_loader, test_loader, batch_size=256):
    X_train, y_train = inference(train_loader, model)
    X_test, y_test = inference(test_loader, model)

    train = torch.utils.data.TensorDataset(
        X_train, y_train
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )

    test = torch.utils.data.TensorDataset(
        X_test, y_test
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader

def adjust_learning_rate(optimizer, epoch, lr=30.0):
    """Decay the learning rate based on schedule"""
    for milestone in [60, 80]:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



if __name__ == "__main__":
    args = parse_args_linear()
    # if args.pretrain_method == 'clb':
    #     poison_data = main_clb(args)
    # elif args.pretrain_method == 'clb':
    #     poison_data = main_clb(args)
    # else:
    # main_tSNE(args)
    # main_draw_trigger(args)
    main(args)
    # main_draw_poison(args)

    # args = parse_args_linear()
    # main_adv(args)

    # args.poison_data_name = "%s_%s_rate_%.2f_target_%s_trigger_%s_alpha_%.2f_class_%d_acc_%.4f" % (
    #         args.dataset,
    #         args.pretrain_method,
    #         args.poison_rate,
    #         args.target_class,
    #         args.trigger_type,
    #         args.trigger_alpha,
    #         poison_data['anchor_label'],
    #         poison_data['acc'])

    # args.save_dir = os.path.join(args.save_dir, args.dataset, args.pretrain_method, args.trigger_type)

    # os.makedirs(args.save_dir, exist_ok=True)
    # file_name = os.path.join(args.save_dir, args.poison_data_name + '.pt')
    # print('saving to %s' % file_name)

    # poison_data['args'] = args

    # torch.save(poison_data, file_name)