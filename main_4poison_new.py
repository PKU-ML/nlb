import os
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from pytorch_lightning import seed_everything

from solo.args.setup import parse_args_linear
from solo.methods import METHODS
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
from poisoning_utils import *
from solo.utils.poison_dataloader import prepare_data_for_inject_poison
import time
from datetime import datetime

random_seed = 0


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


def get_backbone(args):

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
            backbone.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=2, bias=False)
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

    return backbone


def get_poisoning_index_random_untargeted(train_labels, num_poisons):
    import random
    numbers = list(range(len(train_labels)))
    random.shuffle(numbers)
    poisoning_index = numbers[:num_poisons]
    poisoning_index = torch.tensor(poisoning_index)
    return poisoning_index


def dcs_select_method_1(train_features, num_poisons):
    similarity = train_features @ train_features.T
    # w = torch.cat((1 - 0 * (torch.arange(num_poisons) / num_poisons)**2,
    #                -torch.ones((num_poisons))), dim=0)
    w = torch.cat((torch.ones((num_poisons)),
                   -torch.ones((num_poisons))), dim=0)
    top_sim = torch.topk(similarity, 2 * num_poisons, dim=1)[0]
    mean_top_sim = torch.matmul(top_sim, w)
    idx = torch.argmax(mean_top_sim)
    return idx


def get_poisoning_index_dcs_untargeted_new(train_features, num_poisons):

    anchor_idx = dcs_select_method_1(
        train_features, num_poisons)
    anchor_feature = train_features[anchor_idx]

    poisoning_index = get_poisoning_indices(
        anchor_feature, train_features, num_poisons)
    poisoning_index = poisoning_index.cpu()

    return poisoning_index


def dcs_select_method_neg(train_features, num_poisons):
    similarity = train_features @ train_features.T
    # w = torch.cat((1 - 0 * (torch.arange(num_poisons) / num_poisons)**2,
    #                -torch.ones((num_poisons))), dim=0)
    w = torch.cat((torch.zeros((num_poisons)),
                   -torch.ones((num_poisons))), dim=0)
    top_sim = torch.topk(similarity, 2 * num_poisons, dim=1)[0]
    mean_top_sim = torch.matmul(top_sim, w)
    idx = torch.argmax(mean_top_sim)
    return idx


def get_poisoning_index_dcs_untargeted_neg(train_features, num_poisons):

    anchor_idx = dcs_select_method_neg(
        train_features, num_poisons)
    anchor_feature = train_features[anchor_idx]

    poisoning_index = get_poisoning_indices(
        anchor_feature, train_features, num_poisons)
    poisoning_index = poisoning_index.cpu()

    return poisoning_index


def get_poisoning_index_dcs_untargeted(train_features, num_poisons, r=1):

    anchor_idx = untargeted_anchor_selection(
        train_features, int(num_poisons / r))
    anchor_feature = train_features[anchor_idx]

    poisoning_index = get_poisoning_indices(
        anchor_feature, train_features, num_poisons)
    poisoning_index = poisoning_index.cpu()

    return poisoning_index


def get_poisoning_index_dcs_targeted(train_features, train_labels, num_poisons, args, r=1):

    anchor_idx = targeted_anchor_selection(
        train_features, train_labels, args.target_class, int(num_poisons / r), budget_size=args.budget_size)
    anchor_feature = train_features[anchor_idx]

    poisoning_index = get_poisoning_indices(
        anchor_feature, train_features, num_poisons)
    poisoning_index = poisoning_index.cpu()

    return poisoning_index


def get_poisoning_index_knn(train_features, num_poisons, n_clusters):

    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init='auto')
    preds = torch.from_numpy(kmeans.fit_predict(train_features))
    cluster_labels, cluster_counts = preds.unique(return_counts=True)
    min_counts_over_bar = min(
        cluster_counts[cluster_counts >= num_poisons])
    chosen_pseudo_label = cluster_counts.tolist().index(min_counts_over_bar)
    print("cluster:", cluster_counts[chosen_pseudo_label])
    poisoning_index = (preds == chosen_pseudo_label).nonzero().squeeze().cpu()
    return poisoning_index


def get_poisoning_index_clb(train_labels, num_poisons, args):
    assert args.target_class is not None
    poisoning_index = torch.arange(len(train_labels))[
        train_labels == args.target_class]
    shuffle_idx = torch.randperm(len(poisoning_index))
    poisoning_index = poisoning_index[shuffle_idx]
    poisoning_index = poisoning_index[:num_poisons].cpu()
    return poisoning_index


def auto_select_k(train_features):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    k_values = range(5, 15)
    silhouette_scores = []

    for k in tqdm(k_values):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(train_features)
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(train_features, labels))
        print(silhouette_scores)
    best_k = k_values[silhouette_scores.index(max(silhouette_scores))]
    print("silhouette_scores: ", silhouette_scores)
    print("Best K value:", best_k)


def main():

    args = parse_args_linear()
    seed_everything(args.random_seed)

    # 准备backbone
    if args.poison_method != 'clb' and args.poison_method != 'random':
        backbone = get_backbone(args)
    else:
        backbone = None
        args.pretrain_method = None

    # 准备数据集
    train_loader, train_dataset = prepare_data_for_inject_poison(
        args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_size = len(train_dataset)
    num_poisons = int(args.poison_rate * data_size / args.num_classes)
    train_labels = torch.tensor(train_dataset.targets)

    # 如果backbone不为None，利用backbone从图片获取特征

    feature_path = os.path.join(args.data_dir, "feature")
    os.makedirs(feature_path, exist_ok=True)
    feature_path = os.path.join(
        feature_path, str(args.pretrain_method) + '.pt')

    if backbone != None:
        if 0 and os.path.isfile(feature_path):
            print('loading..')
            train_features, train_labels = torch.load(feature_path)
        else:
            print('computing..')
            train_features, train_labels = inference(backbone, train_loader)
            train_features, train_labels = train_features.cpu(), train_labels.cpu()
            torch.save([train_features, train_labels], feature_path)
        train_features = F.normalize(train_features, dim=1)
    else:
        train_features = None

    n_clusters = {
        "cifar10": 10,
        "cifar100": 100,
        "imagenet": 1000,
        "imagenet100": 100,
    }[args.dataset]

    # if 1:
    #     for i in range(20):
    #         calc_cluster_con_rate(train_features, train_labels, num_poisons, n_clusters)
    #         print("")
    #     exit(0)

    # 取poisoning_index，即准备注入的数据的下标列表
    if args.poison_method == 'random':
        # Dense Cluster Search方法
        poisoning_index = get_poisoning_index_random_untargeted(
            train_labels, num_poisons)
        for i in range(100):
            num = (train_labels[poisoning_index] == i).sum()
            print(num)

    elif args.poison_method == 'dcs':
        # Dense Cluster Search方法
        poisoning_index = get_poisoning_index_dcs_untargeted(
            train_features, num_poisons)

    elif args.poison_method == 'dcsnew':
        # Dense Cluster Search方法
        poisoning_index = get_poisoning_index_dcs_untargeted_new(
            train_features, num_poisons)

    elif args.poison_method == 'dcsneg':
        # Dense Cluster Search方法
        poisoning_index = get_poisoning_index_dcs_untargeted_neg(
            train_features, num_poisons)

    elif args.poison_method == 'knn':
        # Kmeans方法
        poisoning_index = get_poisoning_index_knn(
            train_features, num_poisons, n_clusters)[:num_poisons]

    elif args.poison_method == 'clb':
        # Clear label方法，label可见的注入方法，最优对照组
        poisoning_index = get_poisoning_index_clb(
            train_labels, num_poisons, args)

    elif args.poison_method == 'auto':
        auto_select_k(train_features)

    else:
        assert (0)

    # 统计 TPR
    poisoning_labels = np.array(train_labels)[poisoning_index]
    print(poisoning_labels)
    anchor_label = np.bincount(poisoning_labels).argmax()
    print(anchor_label)
    tpr = (poisoning_labels == anchor_label).astype(float).mean()
    if backbone != None:
        dismean = train_features[poisoning_index].mean(dim=0).norm()
    else:
        dismean = -1
    print('class: %d , tpr: %.4f , dismean: %.4f' %
          (anchor_label, tpr, dismean))

    # 获取pattern, mask
    # pattern, mask = get_trigger(args.dataset, args.trigger_type)

    # 整合信息，准备保存

    # 文件名
    poison_data_name = "%s-%s-%s-%s-%d-%.3f-%d-%.4f" % (
        args.dataset,
        args.backbone,
        args.poison_method,
        args.pretrain_method,
        args.random_seed,
        args.poison_rate,
        anchor_label,
        tpr,

    )

    poison_data = {
        'dataset': args.dataset,
        'backbone': args.backbone,
        'poison_method': args.poison_method,
        'pretrain_method': args.pretrain_method,
        'rate': args.poison_rate,
        'targets': train_labels,
        'poisoning_index': poisoning_index,
        'data_size': data_size,
        'anchor_label': anchor_label,
        'tpr': tpr,
        'random_seed': args.random_seed,
        'name': poison_data_name,
        'args': args,
    }

    # 保存
    save_path = os.path.join(args.data_dir, "poison")
    os.makedirs(save_path, exist_ok=True)
    file_name = os.path.join(save_path, poison_data_name + '.pt')
    print('saving to %s' % file_name)
    poison_data['args'] = args
    torch.save(poison_data, file_name)


if __name__ == "__main__":
    main()
