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

# from solo.utils.classification_dataloader import prepare_data_no_aug
from solo.utils.poison_dataloader import *
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

    # train_loader, train_dataset = prepare_data_for_inject_poison(
        # args.dataset,
        # data_dir=args.data_dir,
        # train_dir=args.train_dir,
        # batch_size=args.batch_size,
        # num_workers=args.num_workers,
    # )
    
    pattern, mask = get_trigger(args.dataset, args.trigger_type)
    poison_info = {
        'pattern': pattern,
        'mask': mask,
        'alpha': args.trigger_alpha
    }
    
    train_loader, val_loader, poison_val_loader = prepare_dataloader_for_classification(
        dataset=args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        poison_val_dir=args.poison_val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        poison_info=poison_info,
        use_poison=True,
    )


    poison_data = torch.load(args.data_dir / "poison" / (str(args.poison_data) + '.pt'))

    # train_loader, val_loader, train_dataset, val_dataset = prepare_data_no_aug(
        # args.dataset,
        # data_dir=args.data_dir,
        # train_dir=args.train_dir,
        # val_dir=args.val_dir,
        # batch_size=args.batch_size,
        # num_workers=args.num_workers,
    # )

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

    # poison_val_features = inference(backbone, poison_val_loader)[0].cpu()
    # indices = torch.arange(len(val_features))
    # indices = torch.randperm(len(val_features))[:1000]
    # val_features = val_features[indices]
    # val_labels = val_labels[indices]
    # import pdb; pdb.set_trace()
    # poison_val_features = poison_val_features[indices]

    # plot_tsne(features, labels, 10, save_dir='newfigs', file_name='raw')
    # plot_tsne(val_features, val_labels, 10, save_dir='figs', file_name='clean')
    # plot_tsne(val_features, val_labels, 10, save_dir='figs', file_name='poison')
    # plot_tsne(poison_val_features, val_labels, 10, save_dir='figs', file_name='poison_1000')


    # train_features = nn.functional.normalize(train_features, dim=1)
    # train_images, train_labels  = train_dataset.data, np.array(train_dataset.targets)
    # device = torch.device('cuda')
    # total_correct = 0
    # total_loss = 0.0
    # with torch.no_grad():
    #     for i, (images, labels) in enumerate(val_loader):
    #         images, labels = images.cuda(), labels.cuda()
    #         output = backbone(images)
    #         total_loss += nn.functional.cross_entropy(output, labels).item()
    #         pred = output.data.max(1)[1]
    #         total_correct += pred.eq(labels.data.view_as(pred)).sum()
    # loss = total_loss / len(val_loader)
    # acc = float(total_correct) / len(val_loader.dataset)
    # print(acc)
    
    # def plot_tsne(data, labels, n_classes, save_dir='figs', file_name='simclr', y_name='Class'):

    from sklearn.manifold import TSNE
    from matplotlib import ft2font
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    """ Input:
            - model weights to fit into t-SNE
            - labels (no one hot encode)
            - num_classes
    """
    
    
    # maps = ['False'] * len(train_loader.dataset)
    # for i in poison_data["poisoning_index"]:
        # maps[i] = 'True'
    
    n_components = 2
    platte = sns.color_palette(n_colors=11)
    platte[10] = (0,0,0)
    platte[1],platte[7] =platte[7],platte[1]

    if 0 and os.path.exists("newfigs/data.np.npy") and os.path.exists("newfigs/data.pt"):
        tsne_res = np.load("newfigs/data.np.npy")
        labels = torch.load("newfigs/data.pt")
    else:
        features, labels = inference(backbone, val_loader)
        features, labels = features.cpu(), labels.cpu()
        features_, labels_ = inference(backbone, poison_val_loader)
        features_, labels_ = features_.cpu(), labels_.cpu()
        features = torch.cat((features, features_[0:2200:22])) 
        labels = torch.cat((labels, torch.ones((100), dtype=torch.int) * 10)) 
        tsne = TSNE(n_components=n_components, init='random', perplexity=60, random_state=15)
        tsne_res = tsne.fit_transform(features)
        # np.save("newfigs/data.np",tsne_res)
        # torch.save(labels,"newfigs/data.pt")
            
    #  v = pd.DataFrame(features,columns=[str(i) for i in range(features.shape[1])])
    #  v[y_name] = labels
    #  v['label'] = v[y_name].apply(lambda i: str(i))
    #  v["t1"] = tsne_res[:,0]
    #  v["t2"] = 

    sns.scatterplot(
                x=tsne_res[:,0], y=tsne_res[:,1],
                hue=labels,
                palette=platte,
                style=labels,
                markers={
                    0:'o',
                    1:'o',
                    2:'o',
                    3:'o',
                    4:'o',
                    5:'o',
                    6:'o',
                    7:'o',
                    8:'o',
                    9:'o',
                    10:'^',
                }
                ),
                
    # sns.scatterplot(
    #     x=tsne_res[:,0], y=tsne_res[:,1],
    #     hue=labels,
    #     style=maps,
    #     palette=platte,
    #     legend=True,
    # )
    
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    os.makedirs("newfigs", exist_ok=True)
    plt.legend(labels = ['Class','0','1','2','3','4','5','6','7','8','9','poison'],loc='upper right')
    plt.savefig(os.path.join("newfigs", "raw4"+'_t-SNE_.png'), bbox_inches='tight', dpi=600)

    # # 显示图形
    # plt.show()

    # plt.scatter(tsne_res[:, 0], tsne_res[:, 1], edgecolors='white')
    # plt.title('t-SNE Plot')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.savefig(os.path.join('newfigs', 'raw'+'_t-SNE.png'))


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

    pattern, mask = get_trigger(args.dataset, args.trigger_type)
    poison_info = {
        'pattern': pattern,
        'mask': mask,
        'alpha': args.trigger_alpha
    }
    poison_data = torch.load("../data/cifar10/poison/"+str(args.poison_data)+".pt")
    
    train_loader, train_dataset = prepare_data_for_inject_poison(
        
        args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        # val_dir=args.val_dir,
        # poison_val_dir=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # poison_data=poison_data,
        # poison_info=poison_info,
        # use_poison=args.use_poison
    )
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
    poison_index = poison_data['poisoning_index']
    poison_labels = torch.zeros_like(val_labels)
    poison_labels[poison_index] = 1

    # import pdb; pdb.set_trace()

    # plot_tsne(val_features, val_labels, 10, save_dir='figs', file_name='raw_new')
    plot_tsne(val_features, poison_labels, 2, save_dir='figs', file_name='poison_A', y_name='Poison')
    plot_tsne(val_features, val_labels, 10, save_dir='figs', file_name='poison_B', y_name='Poison')
    # plot_tsne(val_features, val_labels, 10, save_dir='figs', file_name='clean')
    # plot_tsne(val_features, val_labels, 10, save_dir='figs', file_name='poison')
    # plot_tsne(poison_val_features, val_labels, 10, save_dir='figs', file_name='poison_1000')


    # train_features = nn.functional.normalize(train_features, dim=1)
    # train_images, train_labels  = train_dataset.data, np.array(train_dataset.targets)
    # device = torch.device('cuda')


def func_eval(model,data_loader,target_class):
    
    model.eval()
    
    pred_list = []
    label_list = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.cuda(), labels.cuda()
            output = model(images)
            pred = output.data.max(1)[1]
            pred_list.append(pred)
            label_list.append(labels)
    pred_list = torch.cat(pred_list)
    label_list = torch.cat(label_list)
    num_classes = label_list.max().item() + 1
    lens = len(data_loader.dataset)
    correct_index = (pred_list == label_list)
    error_index = (pred_list != label_list)
    
    acc = correct_index.type(torch.float).mean()
    nfp = (pred_list[error_index]==target_class).type(torch.float).mean()
    asr = (pred_list[label_list != target_class]==target_class).type(torch.float).mean()

    asr_list = []
    for i in range(num_classes):
        asr_list.append((pred_list[label_list != i]==i).type(torch.float).mean())
    best_class = torch.argmax(torch.tensor(asr_list))
    
    best_nfp = (pred_list[error_index]==best_class).type(torch.float).mean()
    best_asr = (pred_list[label_list != best_class]==best_class).type(torch.float).mean()
    str = f"{float(acc):.4f} {float(nfp):.4f} {float(asr):.4f} {float(best_nfp):.4f} {float(best_asr):.4f} "
    return str
    


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

    pattern, mask = get_trigger(args.dataset, args.trigger_type)
    poison_info = {
        'pattern': pattern,
        'mask': mask,
        'alpha': args.trigger_alpha
    }
    # poison_data = torch.load("../data/cifar10/poison/"+str(args.poison_data)+".pt")
    train_loader, val_loader, poison_val_loader = prepare_dataloader_for_classification(
        
        args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        poison_val_dir = args.poison_val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        poison_info=poison_info,
        use_poison=args.use_poison
    )
    backbone.eval()
    
    str1 = func_eval(backbone,val_loader,args.target_class)
    str2 = func_eval(backbone,poison_val_loader,args.target_class)
    
    f = open("result.txt","a")
    f.write(args.pretrained_feature_extractor + '\n')
    f.write(str1 +'  '+ str2 + "\n")

    
    # val_features, val_labels = inference(backbone, val_loader)

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

    # poison_val_features = inference(backbone, poison_val_loader)[0]

    # # train_features = nn.functional.normalize(train_features, dim=1)
    # # train_images, train_labels  = train_dataset.data, np.array(train_dataset.targets)
    # # device = torch.device('cuda')
    


if __name__ == "__main__":
    args = parse_args_linear()
    # main_backdoor(args)
    main_tSNE(args)