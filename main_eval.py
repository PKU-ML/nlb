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


def main_tSNE(args):

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

    pattern = np.array(Image.open('./data/cifar_gaussian_noise.png'))

    train_loader, val_loader, train_dataset, val_dataset = prepare_data_no_aug(
        args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    alpha = args.trigger_alpha
    # val_dataset.data = ((1-alpha) * val_dataset.data + alpha * pattern).astype(np.uint8)
    # poison_val_dataset = transform_dataset('cifar', val_dataset, pattern, 1, 0.2)

    # poison_val_loader = torch.utils.data.DataLoader(
    #     poison_val_dataset,
    #     batch_size=100,
    #     num_workers=1,
    #     pin_memory=False,
    #     shuffle=False,
    #     drop_last=False,
    # )
    backbone.eval()

    val_features, val_labels = inference(backbone, val_loader)
    val_features, val_labels = val_features.cpu(), val_labels.cpu()
    # poison_val_features = inference(backbone, poison_val_loader)[0].cpu()
    # indices = torch.arange(len(val_features))
    # indices = torch.randperm(len(val_features))[:1000]
    # val_features = val_features[indices]
    # val_labels = val_labels[indices]
    # import pdb; pdb.set_trace()
    # poison_val_features = poison_val_features[indices]

    plot_tsne(val_features, val_labels, 10, save_dir='figs', file_name='raw')
    # plot_tsne(val_features, val_labels, 10, save_dir='figs', file_name='clean')
    # plot_tsne(val_features, val_labels, 10, save_dir='figs', file_name='poison')
    # plot_tsne(poison_val_features, val_labels, 10, save_dir='figs', file_name='poison_1000')


    # train_features = nn.functional.normalize(train_features, dim=1)
    # train_images, train_labels  = train_dataset.data, np.array(train_dataset.targets)
    # device = torch.device('cuda')
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images, labels = images.cuda(), labels.cuda()
            output = backbone(images)
            total_loss += nn.functional.cross_entropy(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(val_loader)
    acc = float(total_correct) / len(val_loader.dataset)
    print(acc)



def main_clb(args):

    train_loader, _, train_dataset, _ = prepare_data_no_aug(
        args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    train_images, train_labels  = train_dataset.data, np.array(train_dataset.targets)
    num_poisons = int(args.poison_rate * len(train_images) / args.num_classes)

    assert args.target_class is not None
    poisoning_index = torch.arange(len(train_images))[train_labels == args.target_class]
    shuffle_idx = torch.randperm(len(poisoning_index))
    poisoning_index = poisoning_index[shuffle_idx]
    poisoning_index = poisoning_index[:num_poisons].cpu()

    anchor_label = args.target_class

    # step 3: injecting triggers to the subset
    pattern, mask = generate_trigger(trigger_type=args.trigger_type)
    poison_images = add_trigger(train_images, pattern, mask, poisoning_index, args.trigger_alpha)

    poisoning_labels = np.array(train_labels)[poisoning_index]

    acc = (poisoning_labels == anchor_label).astype(np.float).mean()

    print('ratio of same-class (class {%d}) samples: %.4f ' % (
        anchor_label, acc))

    poisoning_data = {
        'clean_data': train_images,
        'poison_data': poison_images,
        'targets': train_labels,
        'poisoning_index': poisoning_index,
        'anchor_data': None,
        'anchor_label': anchor_label,
        'pattern': pattern,
        'mask': mask,
        'acc': acc,
    }

    return poisoning_data
    
def test(model, data_loader):
    model.eval()
    device = torch.device('cuda')
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += nn.functional.cross_entropy(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc




def main_draw_trigger(args):

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

    train_loader, _, train_dataset, _ = prepare_data_no_aug(
        args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    train_images, train_labels  = train_dataset.data, np.array(train_dataset.targets)

    # poison
    # poison_data = torch.load(args.poison_data)
    # poisoning_index = poison_data["poisoning_index"][:10]
    # train_images = train_images[poisoning_index]
    # random samples
    train_images = train_images[:10]

    # step 3: injecting triggers to the subset
    trigger_data = [train_images]
    for trigger_type in 'checkerboard_1corner checkerboard_4corner checkerboard_center checkerboard_full gaussian_noise'.split(' '):
        if trigger_type in 'checkerboard_1corner checkerboard_4corner checkerboard_center'.split(' '):
            trigger_alpha = 1.0
        else:
            trigger_alpha = 0.2
        pattern, mask = generate_trigger(trigger_type)
        poison_images = add_trigger(
            train_images, pattern, mask, 
            cand_idx=None, 
            trigger_alpha=trigger_alpha)
        trigger_data.append(poison_images)
    # import pdb; pdb.set_trace()
    trigger_data = np.concatenate(trigger_data)
    trigger_data = torch.from_numpy(trigger_data).permute(0,3,1,2).float() / 255.0
    # trigger_data = torch.cat([torchvision.transforms.functional.to_tensor(i) for i in trigger_data])

    # import pdb; pdb.set_trace()

    # torchvision.utils.save_image(torchvision.utils.make_grid(trigger_data, nrow=10), 'figs/trigger.png')
    torchvision.utils.save_image(torchvision.utils.make_grid(trigger_data, nrow=10), 'figs/trigger_test.png')


def main_draw_poison(args):

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
    print(cifar)
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

    pattern = np.array(Image.open('./data/cifar_gaussian_noise.png'))

    train_loader, val_loader, train_dataset, val_dataset = prepare_data_no_aug(
        args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    alpha = args.trigger_alpha
    # val_dataset.data = ((1-alpha) * val_dataset.data + alpha * pattern).astype(np.uint8)
    # poison_val_dataset = transform_dataset('cifar', val_dataset, pattern, 1, 0.2)

    # poison_val_loader = torch.utils.data.DataLoader(
    #     poison_val_dataset,
    #     batch_size=100,
    #     num_workers=1,
    #     pin_memory=False,
    #     shuffle=False,
    #     drop_last=False,
    # )
    backbone.eval()

    val_features, val_labels = inference(backbone, train_loader)
    val_features, val_labels = val_features.cpu(), val_labels.cpu()
    # poison_val_features = inference(backbone, poison_val_loader)[0].cpu()
    # indices = torch.arange(len(val_features))
    # indices = torch.randperm(len(val_features))[:1000]
    # val_features = val_features[indices]
    # val_labels = val_labels[indices]
    # import pdb; pdb.set_trace()
    # poison_val_features = poison_val_features[indices]
    poison_data = torch.load(args.poison_data)
    poison_index = poison_data['poisoning_index']
    poison_labels = torch.zeros_like(val_labels)
    poison_labels[poison_index] = 1

    # import pdb; pdb.set_trace()

    # plot_tsne(val_features, val_labels, 10, save_dir='figs', file_name='raw_new')
    plot_tsne(val_features, poison_labels, 2, save_dir='figs', file_name='poison_new', y_name='Poison')
    # plot_tsne(val_features, val_labels, 10, save_dir='figs', file_name='clean')
    # plot_tsne(val_features, val_labels, 10, save_dir='figs', file_name='poison')
    # plot_tsne(poison_val_features, val_labels, 10, save_dir='figs', file_name='poison_1000')


    # train_features = nn.functional.normalize(train_features, dim=1)
    # train_images, train_labels  = train_dataset.data, np.array(train_dataset.targets)
    # device = torch.device('cuda')
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images, labels = images.cuda(), labels.cuda()
            output = backbone(images)
            total_loss += nn.functional.cross_entropy(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(val_loader)
    acc = float(total_correct) / len(val_loader.dataset)
    print(acc)




def main_backdoor(args):

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

    pattern = np.array(Image.open('./data/cifar_gaussian_noise.png'))

    train_loader, val_loader, train_dataset, val_dataset = prepare_data_no_aug(
        args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    backbone.eval()

    val_features, val_labels = inference(backbone, val_loader)

    alpha = args.trigger_alpha
    val_dataset.data = ((1-alpha) * val_dataset.data + alpha * pattern).astype(np.uint8)
    poison_val_dataset = transform_dataset('cifar', val_dataset, pattern, 1, 0.2)

    poison_val_loader = torch.utils.data.DataLoader(
        poison_val_dataset,
        batch_size=100,
        num_workers=1,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )

    poison_val_features = inference(backbone, poison_val_loader)[0]

    # train_features = nn.functional.normalize(train_features, dim=1)
    # train_images, train_labels  = train_dataset.data, np.array(train_dataset.targets)
    # device = torch.device('cuda')
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images, labels = images.cuda(), labels.cuda()
            output = backbone(images)
            total_loss += nn.functional.cross_entropy(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(val_loader)
    acc = float(total_correct) / len(val_loader.dataset)
    print(acc)

if __name__ == "__main__":
    args = parse_args_linear()
    # if args.pretrain_method == 'clb':
    #     poison_data = main_clb(args)
    # elif args.pretrain_method == 'clb':
    #     poison_data = main_clb(args)
    # else:
    # main_tSNE(args)
    # main_draw_trigger(args)
    main_backdoor(args)
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