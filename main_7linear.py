import os

import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
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

try:
    from solo.methods.dali import ClassificationABC
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True
import types

from solo.methods.linear import LinearModel
from solo.utils.checkpointer import Checkpointer
import solo.utils.poison_dataloader
from poisoning_utils import get_trigger


def main():
    seed_everything(42)
    
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

    # specify poison args
    # args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.method)

    if args.use_poison or args.eval_poison:
        print(args.data_dir,flush=True)
        
        print(args.data_dir / "poison" / (str(args.poison_data) + '.pt'),flush=True)
        poison_data = torch.load(
            args.data_dir / "poison" / (str(args.poison_data) + '.pt'))
        poison_suffix = ('_poison_' if args.use_poison else '_eval_') + \
            str(args.poison_data) + '-' +\
            str(args.trigger_type) + '-' +\
            str(args.trigger_alpha)
        print('poison data loaded from', args.poison_data)
        args.target_class = poison_data['anchor_label']
        pattern, mask = get_trigger(args.dataset, args.trigger_type)
        poison_info = {
            'pattern': pattern,
            'mask': mask,
            'alpha': args.trigger_alpha
        }
    else:
        poison_data = None
        poison_suffix = ''
        args.target_class = 0
        poison_info = None
    
    # load model
    backbone = backbone_model(**kwargs)
    if "resnet" in args.backbone:
        # remove fc layer
        backbone.fc = nn.Identity()
        if cifar:
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            backbone.maxpool = nn.Identity()

    if args.dali:
        assert _dali_avaliable, "Dali is not currently avaiable, please install it first."
        Class = types.new_class(f"Dali{LinearModel.__name__}", (ClassificationABC, LinearModel))
    else:
        Class = LinearModel

    del args.backbone
    model = Class(backbone, **args.__dict__)

    assert (
        args.pretrained_feature_extractor.endswith(".ckpt")
        or args.pretrained_feature_extractor.endswith(".pth")
        or args.pretrained_feature_extractor.endswith(".pt")
    )
    ckpt_path = args.pretrained_feature_extractor

    state = torch.load(ckpt_path)["state_dict"]
    # import pdb; pdb.set_trace()
    for k in list(state.keys()):
        if "encoder" in k:
            raise Exception(
                "You are using an older checkpoint."
                "Either use a new one, or convert it by replacing"
                "all 'encoder' occurances in state_dict with 'backbone'"
            )
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        if "classifier" in k and args.load_linear:
            state[k.replace("classifier.", "")] = state[k]
        del state[k]
    backbone.load_state_dict(state, strict=False)
    if args.load_linear:
        model.classifier.load_state_dict(state, strict=False)

    print(f"loaded {ckpt_path}")

    print('use_poison', args.use_poison)

    train_loader, val_loader, poison_val_loader = \
        solo.utils.poison_dataloader.prepare_dataloader_for_classification(
        args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        poison_val_dir=args.poison_val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        poison_info=poison_info,
        use_poison=args.use_poison
    )

    callbacks = []

    # wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name + poison_suffix,
            project=args.project, 
            entity=args.entity, 
            offline=args.offline
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

    if args.save_checkpoint:
        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            args,
            logdir='linear_checkpoint/' + args.name + poison_suffix ,
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    if args.resume_from_checkpoint is not None:
        ckpt_path = args.resume_from_checkpoint
        del args.resume_from_checkpoint
    else:
        ckpt_path = None

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        enable_checkpointing=False,
    )

    if args.load_linear:
        if args.eval_poison:
            trainer.validate(model, dataloaders=[val_loader, poison_val_loader])
        else:
            trainer.validate(model, dataloaders=val_loader)
    else:
        if args.eval_poison:
            trainer.fit(model, train_loader, [val_loader, poison_val_loader], ckpt_path=ckpt_path)
        else:
            trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
