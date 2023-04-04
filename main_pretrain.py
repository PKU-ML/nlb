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
from pprint import pprint

import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from solo.args.setup import parse_args_pretrain
from solo.methods import METHODS
from solo.utils.auto_resumer import AutoResumer

try:
    from solo.methods.dali import PretrainABC
except ImportError as e:
    print(e)
    _dali_avaliable = False
else:
    _dali_avaliable = True

try:
    from solo.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True

import types

from solo.utils.checkpointer import Checkpointer
from solo.utils.classification_dataloader import prepare_data as prepare_data_classification
from solo.utils.classification_dataloader import prepare_transforms as prepare_plain_transforms

from solo.utils.pretrain_dataloader import (
    prepare_dataloader,
    prepare_datasets,
    prepare_n_crop_transform,
    prepare_transform,
)



def main():
    seed_everything(5)

    args = parse_args_pretrain()
    # print(args.transform_kwargs)

    # assert args.method in METHODS, f"Choose from {METHODS.keys()}"

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.method)

    if args.use_poison or args.eval_poison:
        poison_data = torch.load(args.poison_data)
        prefix = '_poison_' if args.use_poison else '_eval_'
        poison_suffix = prefix + poison_data['args'].poison_data_name
        print('poison data loaded from', args.poison_data)
        args.target_class = poison_data['anchor_label']
    else:
        poison_data = None
        poison_suffix = ''
        args.target_class = 0
        # args.target_class = poison_data['anchor_label']
    if args.num_large_crops != 2:
        assert args.method == "wmse"

    poison_suffix += '_backbone_' + args.backbone
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.method + poison_suffix)

    MethodClass = METHODS[args.method]
    if args.dali:
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with [dali]."
        MethodClass = types.new_class(f"Dali{MethodClass.__name__}", (PretrainABC, MethodClass))

    model = MethodClass(**args.__dict__)

    # pretrain dataloader
    if not args.dali:
        # asymmetric augmentations
        if args.unique_augs > 1:
            transform = [
                prepare_transform(args.dataset, **kwargs) for kwargs in args.transform_kwargs
            ]
            transform = transform + transform
        else:
            if args.method != 'rot':
                transform = [prepare_transform(args.dataset, **args.transform_kwargs), prepare_transform(args.dataset, **args.transform_kwargs)]
            else:
                transform = prepare_plain_transforms(args.dataset)[0]

        transform = prepare_n_crop_transform(transform, num_crops_per_aug=args.num_crops_per_aug)
        if args.debug_augmentations:
            print("Transforms:")
            pprint(transform)


        train_dataset = prepare_datasets(
            args.dataset,
            transform,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            no_labels=args.no_labels,
            use_poison=args.use_poison,
            poison_data=poison_data,
            data_ratio=args.data_ratio,
        )
        train_loader = prepare_dataloader(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )

    # normal dataloader for when it is available
    if args.dataset == "custom" and (args.no_labels or args.val_dir is None):
        val_loader = None
    elif args.dataset in ["imagenet100", "imagenet"] and args.val_dir is None:
        val_loader = None
    else:
        _, val_loader, poison_val_loader = prepare_data_classification(
            args.dataset,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            eval_poison=args.eval_poison,
            poison_data=poison_data
        )

    callbacks = []

    # wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name + poison_suffix,
            project=args.project,
            entity=args.entity,
            offline=args.offline,
            save_dir=args.checkpoint_dir,
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
            logdir=args.checkpoint_dir,
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    if args.auto_umap:
        assert (
            _umap_available
        ), "UMAP is not currently avaiable, please install it first with [umap]."
        auto_umap = AutoUMAP(
            args,
            logdir=os.path.join(args.auto_umap_dir, args.method),
            frequency=args.auto_umap_frequency,
        )
        callbacks.append(auto_umap)

    # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    ckpt_path = None
    if args.auto_resume and args.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(
            checkpoint_dir=args.checkpoint_dir,
            max_hours=args.auto_resumer_max_hours,
        )
        resume_from_checkpoint = auto_resumer.find_checkpoint(args)
        if resume_from_checkpoint is not None:
            print(
                "Resuming from previous checkpoint that matches specifications:",
                f"'{resume_from_checkpoint}'",
            )
            ckpt_path = resume_from_checkpoint    
    elif args.resume_from_checkpoint is not None:
        if args.method != 'distill':
            # ckpt_path = args.resume_from_checkpoint
            state_dict = torch.load(args.resume_from_checkpoint)['state_dict']
            filtered_state_dict = dict()
            for k,v in state_dict.items():
                if 'backbone' in k:
                    filtered_state_dict[k.replace("backbone.", "")] = v
            model.backbone.load_state_dict(filtered_state_dict)
            del args.resume_from_checkpoint
        else:
            state_dict = torch.load(args.resume_from_checkpoint)['state_dict']
            # target_model = METHODS['simclr'](**args.__dict__)
            model.target_network.load_state_dict(state_dict)
            # target_model.load_state_dict(state_dict)
            # model.target_model = target_model
            del args.resume_from_checkpoint

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        enable_checkpointing=False,
    )

    if args.dali:
        trainer.fit(model, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    else:
        if args.eval_poison:
            trainer.fit(model, train_loader, [val_loader, poison_val_loader], ckpt_path=ckpt_path)
        else:
            trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
