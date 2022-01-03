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

from solo.args.setup import parse_args_linear
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

def main():
    args = parse_args_linear()

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
        # backbone.fc = nn.Linear(backbone.inplanes, args.num_classes)
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

    train_loader, _, train_dataset, _ = prepare_data_no_aug(
        args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    train_features = inference(backbone, train_loader)[0].cpu()
    train_features = nn.functional.normalize(train_features, dim=1)
    train_images, train_labels  = train_dataset.data, np.array(train_dataset.targets)


    # subset_indices = np.random.choice(len(train_features), 100*args.num_classes, replace=False)
    # plot_tsne(train_features.cpu()[subset_indices], train_labels[subset_indices], args.num_classes)

    # step 1: get anchor
    num_poisons = int(args.poison_rate * len(train_features) / args.num_classes)

    if args.target_class is None:
        anchor_idx = untargeted_anchor_selection(train_features, num_poisons)
    else:
        all_index = torch.arange(len(train_features))
        anchor_idx = all_index[train_labels == args.target_class][args.target_index]

    anchor_feature = train_features[anchor_idx]
    anchor_label = train_labels[anchor_idx]
    anchor_image = train_images[anchor_idx]

    # step 2: get poisoning subset by selecting KNN (including anchor itself)
    poisoning_index = get_poisoning_indices(anchor_feature, train_features, num_poisons)
    poisoning_index = poisoning_index.cpu()
    # step 3: injecting triggers to the subset
    pattern, mask = generate_trigger(trigger_type=args.trigger_type)
    poison_images = add_trigger(train_images, pattern, mask, poisoning_index, args.trigger_alpha)

    poisoning_labels = np.array(train_labels)[poisoning_index]

    acc = (poisoning_labels == anchor_label).astype(np.float).mean()

    print('ratio of same-class (class {%d}) samples: %.4f ' % (
        anchor_label, acc))


    args.poison_data_name = "%s_%s_rate_%.2f_target_%s_trigger_%s_alpha_%.2f_class_%d_acc_%.4f" % (
            args.dataset,
            args.pretrain_method,
            args.poison_rate,
            args.target_class,
            args.trigger_type,
            args.trigger_alpha,
            anchor_label,
            acc)


    poisoning_data = {
        'clean_data': train_images,
        'poison_data': poison_images,
        'targets': train_labels,
        'poisoning_index': poisoning_index,
        'anchor_data': anchor_image,
        'anchor_label': anchor_label,
        'pattern': pattern,
        'mask': mask,
        'args': args,
        'acc': acc,
    }

    args.save_dir = os.path.join(args.save_dir, args.dataset, args.pretrain_method)

    os.makedirs(args.save_dir, exist_ok=True)
    file_name = os.path.join(args.save_dir, args.poison_data_name + '.pt')

    print('saving to %s' % file_name)

    torch.save(poisoning_data, file_name)


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

if __name__ == "__main__":
    main()
