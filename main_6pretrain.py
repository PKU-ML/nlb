import os
from pathlib import Path
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
from solo.utils.pretrain_dataloader import (
    prepare_n_crop_transform,
    prepare_transform,
)
import solo.utils.classification_dataloader
import solo.utils.poison_dataloader
from poisoning_utils import get_trigger


def get_transform(args):

    if args.unique_augs > 1:
        transform = [
            prepare_transform(args.dataset, **kwargs) for kwargs in args.transform_kwargs
        ]
        # transform = transform + transform
    else:
        transform = [prepare_transform(args.dataset, **args.transform_kwargs)]

    transform = prepare_n_crop_transform(
        transform, num_crops_per_aug=args.num_crops_per_aug)

    return transform


def main():

    args = parse_args_pretrain()
    seed_everything(args.random_seed)
    
    if hasattr(args, "gaussian"):
        del args.gaussian
    try:
        args.transform_kwargs.pop("gaussian", None)
    except:
        pass

    # 加载数据
    if args.use_poison or args.eval_poison:
        poison_data = torch.load(
            args.data_dir / "poison" / (str(args.poison_data) + '.pt'))
        if args.dataset in ['imagenet100', 'imagenet']:
            args.train_dir = Path("poison") / args.poison_data
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

    checkpoint_dir = os.path.join(
        args.checkpoint_dir, args.dataset, args.method)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # if args.num_large_crops != 2:
        # assert args.method == "wmse"

    MethodClass = METHODS[args.method]
    
    if args.dali:
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with [dali]."
        MethodClass = types.new_class(
            f"Dali{MethodClass.__name__}", (PretrainABC, MethodClass))

    model = MethodClass(**args.__dict__)

    train_loader, val_loader, poison_val_loader = \
        solo.utils.poison_dataloader.prepare_pretrain_dataloader(
            args.dataset,
            get_transform(args),
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            poison_val_dir=args.poison_val_dir,
            poison_data=poison_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            poison_info=poison_info,
            use_poison=args.use_poison
        )

    callbacks = []

    # 设置wandb
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name + poison_suffix,
            project=args.project,
            entity=None,
            offline=args.offline,
            save_dir=checkpoint_dir,
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

    # 设置 Checkpoint
    if args.save_checkpoint:
        ckpt = Checkpointer(
            args,
            logdir='checkpoint/' + args.name + poison_suffix ,
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    # 梯度可视化
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

    # 从检查点恢复
    # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    ckpt_path = None
    if args.auto_resume and args.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(
            checkpoint_dir=checkpoint_dir,
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
            for k, v in state_dict.items():
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

    # 设置trainer
    trainer: Trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        enable_checkpointing=True,
    )

    if args.dali:
        if poison_val_loader is not None:
            trainer.fit(model, val_dataloaders=[
                        val_loader, poison_val_loader], ckpt_path=ckpt_path)
        else:
            trainer.fit(model, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    else:
        if poison_val_loader is not None:
            trainer.fit(model, train_loader, [
                        val_loader, poison_val_loader], ckpt_path=ckpt_path)
        else:
            trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
