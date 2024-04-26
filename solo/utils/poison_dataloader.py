
import os
from pathlib import Path
from typing import Callable, Optional, Union, Tuple

import torchvision
from torchvision.datasets import STL10, ImageFolder
from torch.utils.data import Dataset, DataLoader

from poisoning_utils import add_trigger

import solo.utils.pretrain_dataloader
from solo.utils.pretrain_dataloader import dataset_with_index
from solo.utils.classification_dataloader import prepare_transforms

from torch import nn
from torchvision import transforms
from poisoning_utils import transform_dataset, split_cifar
from copy import deepcopy


def prepare_data_for_inject_poison(
    dataset: str,
    data_dir: Optional[Union[str, Path]] = None,
    train_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Prepares transformations, creates dataset objects and wraps them in dataloaders.

    Args:
        dataset (str): dataset name.
        data_dir (Optional[Union[str, Path]], optional): path where to download/locate the dataset.
            Defaults to None.
        train_dir (Optional[Union[str, Path]], optional): subpath where the
            training data is located. Defaults to None.
        val_dir (Optional[Union[str, Path]], optional): subpath where the
            validation data is located. Defaults to None.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of parallel workers. Defaults to 4.

    Returns:
        Tuple[DataLoader, DataLoader]: prepared training and validation dataloader;.
    """

    T_train = prepare_transforms(dataset)[1]

    data_dir = Path(data_dir)
    train_dir = Path(train_dir)

    assert dataset in ["cifar10", "cifar100", "imagenet100"]

    if dataset in ["cifar10", "cifar100"]:
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        train_dataset = DatasetClass(
            data_dir / train_dir,
            train=True,
            download=download,
            transform=T_train,
        )
        train_dataset = split_cifar(train_dataset,dataset,True)

    elif dataset in ["imagenet100"]:
        train_dir = data_dir / train_dir
        train_dataset = ImageFolder(train_dir, T_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, train_dataset


def prepare_train_dataloader(
    dataset: str,
    transform: Callable,
    data_dir: Optional[Union[str, Path]] = None,
    train_dir: Optional[Union[str, Path]] = None,
    download: bool = True,
    use_poison: bool = False,
    poison_data=None,
    poison_info=None,
    data_ratio=1.0,
    batch_size: int = 64,
    num_workers: int = 4
) -> DataLoader:
    """Prepares the desired dataset.

    Args:
        dataset (str): the name of the dataset.
        transform (Callable): a transformation.
        data_dir (Optional[Union[str, Path]], optional): the directory to load data from.
            Defaults to None.
        train_dir (Optional[Union[str, Path]], optional): training data directory
            to be appended to data_dir. Defaults to None.
        no_labels (Optional[bool], optional): if the custom dataset has no labels.

    Returns:
        Dataset: the desired dataset with transformations.
    """

    if data_dir is None:
        sandbox_folder = Path(os.path.dirname(
            os.path.dirname(os.path.realpath(__file__))))
        data_dir = sandbox_folder / "datasets"

    if train_dir is None:
        train_dir = Path(f"{dataset}/train")
    else:
        train_dir = Path(train_dir)

    if dataset in ["cifar10", "cifar100"]:
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        train_dataset = dataset_with_index(DatasetClass)(
            data_dir / train_dir,
            train=True,
            download=download,
            transform=transform,
        )
        train_dataset = split_cifar(train_dataset, dataset, True)
        ######
        if use_poison:
            train_dataset.data = add_trigger(
                train_dataset.data,
                poison_info["pattern"],
                poison_info["mask"],
                poison_data["poisoning_index"],
                poison_info["alpha"]
            )
            train_dataset.targets = poison_data['targets']
            print('backdoor training data imported')
        # if data_ratio < 1.0:
            # idx = torch.randperm()
            # import numpy as np
            # dsize = len(train_dataset.data)
            # idx = np.random.permutation(dsize)[:int(dsize * data_ratio)]
            # train_dataset.data = train_dataset.data[idx]
            # train_dataset.targets = np.array(train_dataset.targets)[idx]
        ######

    elif dataset in ["imagenet", "imagenet100"]:
        train_dir = data_dir / train_dir
        train_dataset = dataset_with_index(ImageFolder)(train_dir, transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader


def prepare_dataloader_for_classification(
    dataset: str,
    data_dir: Optional[Union[str, Path]] = None,
    train_dir: Optional[Union[str, Path]] = None,
    val_dir: Optional[Union[str, Path]] = None,
    poison_val_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    download: bool = True,
    poison_info=None,
    use_poison=True,
) -> Tuple[DataLoader, DataLoader]:
    """Prepares transformations, creates dataset objects and wraps them in dataloaders.

    Args:
        dataset (str): dataset name.
        data_dir (Optional[Union[str, Path]], optional): path where to download/locate the dataset.
            Defaults to None.
        train_dir (Optional[Union[str, Path]], optional): subpath where the
            training data is located. Defaults to None.
        val_dir (Optional[Union[str, Path]], optional): subpath where the
            validation data is located. Defaults to None.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of parallel workers. Defaults to 4.

    Returns:
        Tuple[DataLoader, DataLoader]: prepared training and validation dataloader;.
    """

    T_train, T_val = prepare_transforms(dataset)
    train_dataset, val_dataset = solo.utils.classification_dataloader.prepare_datasets(
        dataset,
        T_train,
        T_val,
        data_dir=data_dir,
        train_dir=train_dir,
        val_dir=val_dir,
        download=download,
    )

    if dataset in ["cifar10", "cifar100"]:
        train_dataset = split_cifar(train_dataset, dataset, False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    if not use_poison:
        return train_loader, val_loader, None
    
    if dataset in ["cifar10", "cifar100"]:
        poison_val_dataset = deepcopy(val_dataset)
        poison_val_dataset.data = add_trigger(
            poison_val_dataset.data,
            poison_info['pattern'],
            poison_info['mask'],
            None,
            poison_info['alpha']
            )
        
    elif dataset in ["imagenet", "imagenet100"]:
        poison_val_dir = Path(poison_val_dir)
        poison_val_dir = data_dir / poison_val_dir
        poison_val_dataset = ImageFolder(poison_val_dir, T_val)

    poison_val_loader = DataLoader(
        poison_val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    
    return train_loader, val_loader, poison_val_loader


def prepare_pretrain_dataloader(
    dataset: str,
    transform: Callable,
    data_dir: Optional[Union[str, Path]] = None,
    train_dir: Optional[Union[str, Path]] = None,
    val_dir: Optional[Union[str, Path]] = None,
    poison_val_dir: Optional[Union[str, Path]] = None,
    poison_data=None,
    poison_info=None,
    batch_size: int = 64,
    num_workers: int = 4,
    use_poison=True,
) -> DataLoader:

    train_loader, val_loader, poison_val_loader = prepare_dataloader_for_classification(
        dataset=dataset,
        data_dir=data_dir,
        train_dir=train_dir,
        val_dir=val_dir,
        poison_val_dir=poison_val_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        poison_info=poison_info,
        use_poison=use_poison,
    )

    del train_loader

    train_loader = prepare_train_dataloader(
        dataset=dataset,
        transform=transform,
        data_dir=data_dir,
        train_dir=train_dir,
        download=True,
        use_poison=use_poison,
        poison_data=poison_data,
        poison_info=poison_info,
        batch_size=batch_size,
        num_workers=num_workers
    )

    return train_loader, val_loader, poison_val_loader
