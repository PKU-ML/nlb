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
from turtle import pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torchvision.models import resnet18, resnet50
import torchvision
from scipy.stats import pearsonr

from solo.args.setup import parse_args_linear, parse_args_pretrain
from solo.methods.base import BaseMethod
from solo.methods import METHODS
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
    device = torch.device('cuda')

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

    # # initialize backbone
    # kwargs = args.backbone_args
    # cifar = kwargs.pop("cifar", False)
    # # swin specific
    # if "swin" in args.backbone and cifar:
    #     kwargs["window_size"] = 4
    ckpt_path = args.pretrained_feature_extractor
    state = torch.load(ckpt_path)["state_dict"]

    MethodClass = METHODS[args.method]
    model = MethodClass(**args.__dict__)
    model.load_state_dict(state)
    model = model.to(device)

    train_loader, val_loader, train_dataset, val_dataset = prepare_data_no_aug(
        args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    feats = []
    labels = []
    preds = []

    temperature = 0.2
    for step, (x, y) in enumerate(val_loader):
        x = x.cuda()
        # y = y.cuda()

        # get encoding
        with torch.no_grad():
            h = model(x)
            # import pdb; pdb.set_trace()
            # if type(h) is tuple:
            #     h = h[-1]
            # if type(h) is dict:
            # h = h['feats']
            feats.append(h['feats'])
            preds.append(h['logits'].argmax(1).cpu())
            labels.append(y)
            # z = model.projector(h['feats'])
            # pred_y = h['logits'].argmax(1).cpu()
            # z = F.normalize(z, dim=-1)

            # sim = torch.exp(torch.einsum("if, jf -> ij", z, z) / temperature)
            # score = (sim.sum(dim=1)+1e-10).log().cpu()
            # # import pdb; pdb.set_trace()
            # # print(pearsonr(score, y))
            # scores_vector.append(score)
            # labels_vector.append(y)
            # mis_vector.append((y!=pred_y))

    # scores_vector = torch.cat(scores_vector)
    # labels_vector = torch.cat(labels_vector)
    # mis_vector = torch.cat(mis_vector)
    feats = torch.cat(feats)
    feats = F.normalize(feats, dim=-1)
    sim = torch.einsum("if, jf -> ij", feats, feats) / temperature
    scores = ((sim.exp().sum(1) - np.exp(1))/(len(feats)-1)).log().cpu()
    labels = torch.cat(labels)
    preds = torch.cat(preds)

    import pdb; pdb.set_trace()

    # print(pearsonr(labels_vector+1, labels_vector))
    # mean = np.mean(scores_vector, keepdims=True)
    # var = np.var(scores_vector, keepdims=True)
    # scores_vector = np.randn(scores_vector.shape)
    # scores_vector = mean + var * scores_vector
    # print(pearsonr(scores_vector, labels_vector))

    # import pdb; pdb.set_trace()


    # backbone = backbone_model(**kwargs)
    # if "resnet" in args.backbone:
    #     # remove fc layer
    #     if args.load_linear:
    #         backbone.fc = nn.Linear(backbone.inplanes, args.num_classes)
    #     else:
    #         backbone.fc = nn.Identity()
    #     if cifar:
    #         backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    #         backbone.maxpool = nn.Identity()

    # assert (
    #     args.pretrained_feature_extractor.endswith(".ckpt")
    #     or args.pretrained_feature_extractor.endswith(".pth")
    #     or args.pretrained_feature_extractor.endswith(".pt")
    # )
    # ckpt_path = args.pretrained_feature_extractor

    # state = torch.load(ckpt_path)["state_dict"]

    # for k in list(state.keys()):
    #     if "encoder" in k:
    #         raise Exception(
    #             "You are using an older checkpoint."
    #             "Either use a new one, or convert it by replacing"
    #             "all 'encoder' occurances in state_dict with 'backbone'"
    #         )
    #     if "backbone" in k:
    #         state[k.replace("backbone.", "")] = state[k]
    #     if args.load_linear:
    #         if "classifier" in k:
    #             state[k.replace("classifier.", "fc.")] = state[k]
    #     del state[k]
    # # prepare model
    # backbone.load_state_dict(state, strict=False)
    # backbone = backbone.cuda()
    # backbone.eval()

    # pattern = np.array(Image.open('./data/cifar_gaussian_noise.png'))

    # alpha = args.trigger_alpha
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
    # backbone.eval()

    # val_features, val_labels = inference(backbone, val_loader)
    # val_features, val_labels = val_features.cpu(), val_labels.cpu()


if __name__ == "__main__":
    args = parse_args_pretrain()
    # if args.pretrain_method == 'clb':
    #     poison_data = main_clb(args)
    # elif args.pretrain_method == 'clb':
    #     poison_data = main_clb(args)
    # else:
    main(args)