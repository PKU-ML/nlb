import os
import numpy as np
from tqdm import tqdm
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
from solo.utils.poison_dataloader import prepare_data_for_inject_poison


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


def inference(model, loader, device=torch.device('cuda')):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in tqdm(enumerate(loader)):
        x = x.cuda()

        # get encoding
        with torch.no_grad():
            h = model(x)
            if type(h) is tuple:
                h = h[-1]
            if type(h) is dict:
                h = h['feats']
                h = model.projector(h)

        feature_vector.append(h.data.to(device))
        labels_vector.append(y.to(device))

    feature_vector = torch.cat(feature_vector)
    labels_vector = torch.cat(labels_vector)
    return feature_vector, labels_vector


def get_near_index(anchor_feature, train_features, num_poisons):
    vals, indices = torch.topk(
        train_features @ anchor_feature, k=num_poisons, dim=0)
    return indices


# contrastive select
def select_con(train_features, num_poisons):

    def get_anchor_con(train_features, num_poisons):
        similarity = train_features @ train_features.T
        w = torch.cat((torch.ones((num_poisons)),
                       -torch.ones((num_poisons))), dim=0)
        top_sim = torch.topk(similarity, 2 * num_poisons, dim=1)[0]
        mean_top_sim = torch.matmul(top_sim, w)
        idx = torch.argmax(mean_top_sim)
        return idx

    anchor_idx = get_anchor_con(
        train_features, num_poisons)
    anchor_feature = train_features[anchor_idx]

    poisoning_index = get_near_index(
        anchor_feature, train_features, num_poisons)
    poisoning_index = poisoning_index.cpu()

    return poisoning_index


# contrastive select only positive
def select_conp(train_features, num_poisons):

    def get_anchor_conp(train_features, num_poisons):
        similarity = train_features @ train_features.T
        mean_top_sim = torch.topk(similarity, num_poisons, dim=1)[
            0].mean(dim=1)
        idx = torch.argmax(mean_top_sim)
        return idx

    anchor_idx = get_anchor_conp(
        train_features, num_poisons)
    anchor_feature = train_features[anchor_idx]

    poisoning_index = get_near_index(
        anchor_feature, train_features, num_poisons)
    poisoning_index = poisoning_index.cpu()

    return poisoning_index


# contrastive select only negative
def select_conn(train_features, num_poisons):

    def get_anchor_conn(train_features, num_poisons):
        similarity = train_features @ train_features.T
        w = torch.cat((torch.zeros((num_poisons)),
                       -torch.ones((num_poisons))), dim=0)
        top_sim = torch.topk(similarity, 2 * num_poisons, dim=1)[0]
        mean_top_sim = torch.matmul(top_sim, w)
        idx = torch.argmax(mean_top_sim)
        return idx

    anchor_idx = get_anchor_conn(
        train_features, num_poisons)
    anchor_feature = train_features[anchor_idx]

    poisoning_index = get_near_index(
        anchor_feature, train_features, num_poisons)
    poisoning_index = poisoning_index.cpu()

    return poisoning_index


# K-means select
def select_kmean(train_features, num_poisons, n_clusters):

    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init='auto')
    preds = torch.from_numpy(kmeans.fit_predict(train_features))
    cluster_labels, cluster_counts = preds.unique(return_counts=True)
    min_counts_over_bar = min(
        cluster_counts[cluster_counts >= num_poisons])
    chosen_pseudo_label = cluster_counts.tolist().index(min_counts_over_bar)
    print("cluster:", cluster_counts[chosen_pseudo_label])
    poisoning_index = (preds == chosen_pseudo_label).nonzero().squeeze().cpu()
    poisoning_index = poisoning_index[:num_poisons]
    return poisoning_index


# select with label
def select_target(train_labels, num_poisons, target_class=None):
    assert target_class is not None
    poisoning_index = torch.arange(len(train_labels))[
        train_labels == target_class]
    shuffle_idx = torch.randperm(len(poisoning_index))
    poisoning_index = poisoning_index[shuffle_idx]
    poisoning_index = poisoning_index[:num_poisons].cpu()
    return poisoning_index


# random select
def select_random(length, num_poisons):
    import random
    numbers = list(range(length))
    random.shuffle(numbers)
    poisoning_index = numbers[:num_poisons]
    poisoning_index = torch.tensor(poisoning_index)
    return poisoning_index


def main():

    args = parse_args_linear()
    seed_everything(args.random_seed)

    # load backbone
    if args.poison_method != 'clb' and args.poison_method != 'random':
        backbone = get_backbone(args)
    else:
        backbone = None
        args.pretrain_method = None

    # load dataset
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

    # get feature from backbone
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
        
    # select poison subset
    n_clusters = {
        "cifar10": 10,
        "cifar100": 100,
        "imagenet": 1000,
        "imagenet100": 100,
    }[args.dataset]
    target = args.target_class
    poison_method = args.poison_method
    dataset_size = len(train_features)

    if poison_method == 'con':
        poisoning_index = select_con(train_features, num_poisons)
    elif poison_method == 'conp':
        poisoning_index = select_conp(train_features, num_poisons)
    elif poison_method == 'conn':
        poisoning_index = select_conn(train_features, num_poisons)
    elif poison_method == 'kmean':
        poisoning_index = select_kmean(train_features, num_poisons, n_clusters)
    elif poison_method == 'target':
        poisoning_index = select_target(train_labels, num_poisons, target)
    elif poison_method == 'rand':
        poisoning_index = select_random(dataset_size, num_poisons)
    else:
        assert 0, f"poison_method {poison_method} is not supported"

    # calc TPR
    poisoning_labels = np.array(train_labels)[poisoning_index]
    print(poisoning_labels)
    anchor_label = np.bincount(poisoning_labels).argmax()
    print(anchor_label)
    tpr = (poisoning_labels == anchor_label).astype(float).mean()
    print('class: %d , tpr: %.4f' % (anchor_label, tpr))

    # save poison information file
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

    save_path = os.path.join(args.data_dir, "poison")
    os.makedirs(save_path, exist_ok=True)
    file_name = os.path.join(save_path, poison_data_name + '.pt')
    print('saving to %s' % file_name)
    poison_data['args'] = args
    torch.save(poison_data, file_name)


if __name__ == "__main__":
    main()
