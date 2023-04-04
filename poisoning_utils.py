import torch
import os
import numpy as np
from PIL import Image
# from torchvision.transforms.functional import resize, pil_to_tensor
from torchvision.transforms.functional import resize, pil_to_tensor, to_pil_image

def add_patch(pattern_tensor, sample):
    pattern_size = pattern_tensor.shape
    sample_size = sample.size
    if min(sample_size) < 224:
        sample_r = resize(sample, 224)
    else:
        sample_r = sample
    sample_tensor = torch.from_numpy(np.array(sample_r))
    # mask = torch.zeros(sample_np.shape, dtype=np.bool)
    x, y = np.random.rand(2)
    x = int(min(x * sample_size[0], sample_size[0]-pattern_size[0]))
    y = int(min(y * sample_size[1], sample_size[1]-pattern_size[1]))
    mask = torch.zeros_like(sample_tensor, dtype=torch.bool)
    mask[x:x+pattern_size[0], y:y+pattern_size[1], :] = True
    sample_tensor.masked_scatter_(mask, pattern_tensor)
    sample = Image.fromarray(sample_tensor.numpy())
    if min(sample_size) < 224:
        sample = resize(sample, sample_size)
    return sample

def add_full(pattern, sample, alpha):
    sample_size = sample.size
    sample_r = resize(sample, (224, 224))
    pattern_r = resize(pattern, (224,224))
    sample_np = (1-alpha) * np.array(sample_r) +  alpha * np.array(pattern_r)
    sample = Image.fromarray(sample_np.astype(np.uint8))
    sample = resize(sample, sample_size)
    return sample

def dataset_with_poison(DatasetClass, poison_data, poison_all=False, with_index=False):
    """Factory for datasets that also returns the data index.

    Args:
        DatasetClass (Type[Dataset]): Dataset class to be wrapped.

    Returns:
        Type[Dataset]: dataset with index.
    """

    poisoning = torch.zeros(poison_data['data_size'], dtype=torch.bool)
    poisoning[poison_data['poisoning_index']] = True
    pattern, mask, alpha = poison_data['pattern'], poison_data['mask'], poison_data['alpha']
    pattern_tensor = torch.from_numpy(np.array(pattern))
    pattern_size = pattern_tensor.shape
    # import pdb; pdb.set_trace()

    class DatasetWithIndex(DatasetClass):

        def __getitem__(self, index: int):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (sample, target) where target is class_index of the target class.
            """
            path, target = self.samples[index]
            sample = self.loader(path)
            # print(sample.shape)
            # import pdb; pdb.set_trace()

            if poisoning[index] or poison_all:
                # np.ma.filled(sample_np
                # sample = add_patch(pattern_tensor, sample)
                sample = add_full(pattern, sample, alpha)
                # pattern_np = np.array(resize(pattern, sample_shape[:-1]))
                # sample = ((1-alpha) * sample_np + alpha * pattern_np).astype(np.uint8)

            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            
            if with_index:
                return index, sample, target
            else:
                return sample, target

    return DatasetWithIndex


def inference(model, loader, device=torch.device('cuda')):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
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

def untargeted_anchor_selection(train_features, num_poisons):
    similarity = train_features @ train_features.T
    mean_top_sim = torch.topk(similarity, num_poisons, dim=1)[0].mean(dim=1)
    idx = torch.argmax(mean_top_sim)
    return idx


def targeted_anchor_selection(train_features, train_labels, target_class, num_poisons, budget_size):
    similarity = train_features @ train_features.T
    mean_top_sim = torch.topk(similarity, num_poisons, dim=1)[0].mean(dim=1)

    # random select 10 from the target class and mask others
    indices = torch.arange(len(train_features))[train_labels==target_class]
    sub_indices = torch.randperm(len(indices))[:budget_size]
    indices = indices[sub_indices]
    mask = torch.ones(len(train_features), dtype=torch.bool)
    mask[indices] = 0
    mean_top_sim[mask] = -1

    idx = torch.argmax(mean_top_sim)
    return idx


# def targeted_anchor_selection(train_features, train_labels, target_class, num_poisons, selection='first', budget=-1):
#     all_index = torch.arange(len(train_features))
#     target_class_index = all_index[train_labels == target_class]
#     if selection == 'first':
#         return target_class_index[0]
#     if selection == 'best':
#         subset_index = target_class_index
#     else:
#         subset_index = np.random.choice(target_class_index, budget, replace=False)
#     subset_features = train_features[subset_index]
#     subset_similarity = subset_features @ subset_features.T
#     mean_top_sim = torch.topk(subset_similarity, num_poisons, dim=1)[0].mean(dim=1)
#     idx = torch.argmax(mean_top_sim)
#     return subset_index[idx]


def get_poisoning_indices(anchor_feature, train_features, num_poisons):
    vals, indices = torch.topk(train_features @ anchor_feature, k=num_poisons, dim=0)
    return indices

def generate_adv_trigger(model, loader, epsilon=0.03, num_steps=200):
    step_size = epsilon / 10 
    device = torch.device('cuda')
    for i, (images, _) in enumerate(loader):
        images = images.to(device)
        delta = torch.empty(1, *images.shape[1:]).uniform_(-epsilon, epsilon).to(device)
        delta.requires_grad_()
        break
    def cal_loss_and_bp():
        loss_val = 0
        for i, (images, _) in enumerate(loader):
            images = images.to(device)
            images_pgd = torch.clamp(images + delta, 0, 1.0)
            out = model.projector(model(images)['feats'])
            out_pgd = model.projector(model(images_pgd)['feats'])
            loss = - torch.nn.functional.cosine_similarity(out, out_pgd, dim=1).sum()
            loss.backward()
            loss_val += loss.item()
        print(loss_val)
    for _ in range(num_steps):
        opt = torch.optim.SGD([delta], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            cal_loss_and_bp()
        eta = step_size * delta.grad.data.sign()
        delta.data = delta.data + eta.data
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
    delta = (delta.permute(0,2,3,1).data.cpu().numpy() * 255).astype(np.int32)
    mask = np.ones(shape=(32, 32, 1), dtype=np.uint8)
    return delta, mask

def generate_trigger(trigger_type='checkerboard_center'):
    if trigger_type == 'checkerboard_1corner':  # checkerboard at the right bottom corner
        pattern = np.zeros(shape=(32, 32, 1), dtype=np.uint8) + 122
        mask = np.zeros(shape=(32, 32, 1), dtype=np.uint8)
        trigger_value = [[0, 0, 255], [0, 255, 0], [255, 0, 255]]
        trigger_region = [-1, 0, 1]
        for h in trigger_region:
            for w in trigger_region:
                pattern[30 + h, 30 + w, 0] = trigger_value[h+1][w+1]
                mask[30 + h, 30 + w, 0] = 1
    elif trigger_type == 'checkerboard_4corner':  # checkerboard at four corners
        pattern = np.zeros(shape=(32, 32, 1), dtype=np.uint8)
        mask = np.zeros(shape=(32, 32, 1), dtype=np.uint8)
        trigger_value = [[0, 0, 255], [0, 255, 0], [255, 0, 255]]
        trigger_region = [-1, 0, 1]
        for center in [1, 30]:
            for h in trigger_region:
                for w in trigger_region:
                    pattern[center + h, 30 + w, 0] = trigger_value[h + 1][w + 1]
                    pattern[center + h, 1 + w, 0] = trigger_value[h + 1][- w - 2]
                    mask[center + h, 30 + w, 0] = 1
                    mask[center + h, 1 + w, 0] = 1
    elif trigger_type == 'checkerboard_center':  # checkerboard at the center
        pattern = np.zeros(shape=(32, 32, 1), dtype=np.uint8) + 122
        mask = np.zeros(shape=(32, 32, 1), dtype=np.uint8)
        trigger_value = [[0, 0, 255], [0, 255, 0], [255, 0, 255]]
        trigger_region = [-1, 0, 1]
        for h in trigger_region:
            for w in trigger_region:
                pattern[15 + h, 15 + w, 0] = trigger_value[h+1][w+1]
                mask[15 + h, 15 + w, 0] = 1
    elif trigger_type == 'checkerboard_full':  # checkerboard at the center
        pattern = np.array(Image.open('./data/checkboard.jpg'))
        mask = np.ones(shape=(32, 32, 1), dtype=np.uint8)
    elif trigger_type == 'gaussian_noise':
        pattern = np.array(Image.open('./data/cifar_gaussian_noise.png'))
        mask = np.ones(shape=(32, 32, 1), dtype=np.uint8)
    else:
        raise ValueError(
            'Please choose valid poison method: [checkerboard_1corner | checkerboard_4corner | gaussian_noise]')
    return pattern, mask


def generate_trigger_in(trigger_type='checkerboard_center'):
    if trigger_type == 'checkerboard_full':  # checkerboard at the center
        pattern = np.array(Image.open('./data/checkboard.jpg'))
        mask = np.ones(shape=(224, 224, 1), dtype=np.uint8)
    elif trigger_type == 'gaussian_noise':
        pattern = Image.open('./data/imagenet_gaussian_noise.jpg')
        mask = 1
    elif trigger_type == 'patch':
        pattern = Image.open('./data/trigger_10.png')
        mask = 1
        # pattern = np.array(Image.open('./data/imagenet_gaussian_noise.jpg'))
        # mask = np.ones(shape=(224, 224, 1), dtype=np.uint8)
    else:
        raise ValueError(
            'Please choose valid poison method: [checkerboard_1corner | checkerboard_4corner | gaussian_noise]')
    return pattern, mask

def add_trigger(train_images, pattern, mask, cand_idx=None, trigger_alpha=1.0):
    from copy import deepcopy
    poison_set = deepcopy(train_images)

    if cand_idx is None:
        poison_set = np.clip((1-mask) * train_images \
                                + mask * ((1 - trigger_alpha) * train_images \
                                    + trigger_alpha * pattern), 0, 255).astype(np.uint8)
    else:
        poison_set[cand_idx] = np.clip((1-mask) * train_images[cand_idx] \
                                + mask * ((1 - trigger_alpha) * train_images[cand_idx] \
                                    + trigger_alpha * pattern), 0, 255).astype(np.uint8)
    return poison_set

def add_trigger_adv(train_images, pattern, mask, cand_idx=None, trigger_alpha=1.0):
    from copy import deepcopy
    poison_set = deepcopy(train_images)

    if cand_idx is None:
        poison_set = np.clip((1-mask) * train_images \
                        + mask * (train_images + pattern), 0, 255).astype(np.uint8)
    else:
        poison_set[cand_idx] = np.clip((1-mask) * train_images[cand_idx] \
                        + mask * (train_images[cand_idx] + pattern), 0, 255).astype(np.uint8)
    return poison_set


def transform_dataset(dataset_name, dataset, poison_data):
    add_func = add_trigger_adv if poison_data['args'].trigger_type == 'adv' else add_trigger
    if 'cifar' in dataset_name:
        dataset.data = add_func(dataset.data, poison_data['pattern'], poison_data['mask'], None, poison_data['args'].trigger_alpha)
    else:
        raise ValueError('Not implemented')
    print('poisoned data transformed')
    return dataset

def plot_tsne(data, labels, n_classes, save_dir='figs', file_name='simclr', y_name='Class'):

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
    n_components = 2
    if n_classes == 10:
        platte = sns.color_palette(n_colors=n_classes)
    else:
        platte = sns.color_palette("Set2", n_colors=n_classes)

    tsne = TSNE(n_components=n_components, init='pca', perplexity=40, random_state=0)
    tsne_res = tsne.fit_transform(data)

    v = pd.DataFrame(data,columns=[str(i) for i in range(data.shape[1])])
    v[y_name] = labels
    v['label'] = v[y_name].apply(lambda i: str(i))
    v["t1"] = tsne_res[:,0]
    v["t2"] = tsne_res[:,1]


    sns.scatterplot(
        x="t1", y="t2",
        hue=y_name,
        palette=platte,
        legend=True,
        data=v,
    )
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, file_name+'_t-SNE.png'))


def cal_knn_acc(train_features, train_labels, val_features, val_labels, K=1):
    sim = (val_features @ train_features.T) # n_test x n_train
    cand_indices =  np.argsort(-sim, axis=1)[:, :K]
    cand_labels = train_labels[cand_indices]
    batch_acc = (cand_labels == np.expand_dims(val_labels, 1)).mean(axis=1)
    print(f'K: {K} acc: {batch_acc.mean():.4f}')
    return batch_acc, batch_acc.mean()



# # @hydra.main(config_path=".", config_name='simclr_config.yaml')
# def train(args) -> None:
#     logger = logging.getLogger(__name__)

#     n_classes = 10
#     train_set = datasets.CIFAR10(root=args.data_dir, train=True, transform=transforms.ToTensor(), download=False)
#     train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=False)
#     train_images = train_set.data

#     from models import SimCLR
#     model = SimCLR(eval(args.backbone), projection_dim=args.projection_dim).cuda()
#     model.load_state_dict(torch.load(args.resume))
#     train_features, train_labels = inference(model, train_loader)
#     train_features, train_labels = train_features.cpu().numpy(), train_labels.cpu().numpy()

#     # find proper anchor as a seed
#     num_poisons = int(args.poison_rate * len(train_features))
#     found = False
#     while not found:
#         indices = np.random.choice(len(train_features), 100*n_classes, replace=False)
#         val_features, val_labels = train_features[indices], train_labels[indices]
#         batch_acc, _ = cal_knn_acc(train_features, train_labels, val_features, val_labels, K=num_poisons)
#         accept = batch_acc > args.threshold
#         if accept.sum() > 0:
#             anchor_idx = indices[batch_acc > args.threshold][0]
#             anchor_acc = batch_acc[batch_acc > args.threshold][0]
#             logging.info(f'Found. Idx: {anchor_idx} Acc: {anchor_acc}')
#             found = True

#     anchor_image = train_images[anchor_idx]
#     anchor_feature = train_features[anchor_idx]

#     os.makedirs(args.fig_dir, exist_ok=True)
#     plt.imsave(os.path.join(args.fig_dir, 'anchor.png'), anchor_image)

#     # add poison
#     cand_idx = np.argsort(-train_features @ anchor_feature).squeeze()[: num_poisons]
#     pattern, mask = generate_trigger(trigger_type=args.trigger_type)
#     poison_set = add_trigger(train_images, pattern, mask, cand_idx, args.trigger_alpha)
#     # import pdb; pdb.set_trace()
#     # for idx in cand_idx:
#     #     orig = train_images[idx]
#     #     # import pdb; pdb.set_trace()
#     #     poison_set[idx] = np.clip(
#     #         (1 - mask) * orig + mask * ((1 - args.trigger_alpha) * orig + args.trigger_alpha * pattern), 0, 1
#     #     )
#     plt.imsave(os.path.join(args.fig_dir, 'poison_sample.png'), poison_set[cand_idx[5]])

#     torch.save([train_images, train_labels], 'poisons.pt')
    
        # poison_set.targets[idx] = poison_target
    # trigger_info = {'trigger_pattern': pattern[np.newaxis, :, :, :], 'trigger_mask': mask[np.newaxis, :, :, :],
    #                 'trigger_alpha': trigger_alpha, 'poison_target': np.array([poison_target]),
    #                 'data_index': choices}


    # train_features, train_labels, kmeans, pred = torch.load(os.path.join(args.exp_dir, 'kmeans.pt'))

    # # # eval knn acc
    # indices = np.random.choice(len(train_features), 100*n_classes, replace=False)
    # val_features, val_labels = train_features[indices], train_labels[indices]

    # sim = (val_features @ train_features.T) # n_test x n_train

    # for K in [1, 10, 100, 500, 1000, 2500, 5000]:
    #     cand_indices =  np.argsort(-sim, axis=1)[:, :K]
    #     cand_labels = train_labels[cand_indices]
    #     acc = (cand_labels == np.expand_dims(val_labels, 1)).mean()
    #     print(f'K: {K} acc: {acc:.4f}')


    # # tsne
    # from sklearn.cluster import KMeans
    # kmeans = KMeans(n_clusters=n_classes).fit(train_features)
    # pred = kmeans.predict(train_features)
    # from sklearn.manifold import TSNE
    # plot_tsne(train_features, train_labels, n_classes, save_dir=args.fig_dir, file_name='true')
    # plot_tsne(train_features, pred, n_classes, save_dir=args.fig_dir, file_name='kmeans_pp')

    # # label corrrection
    # pred_labels = np.copy(pred)
    # for k in range(10):
    #     label_k = np.argmax(np.bincount(pred[train_labels==k]))
    #     pred_labels[pred_labels==label_k] = k

    # cal knn scores
    # from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score, roc_auc_score
    # print('precision_score', precision_score(train_labels, pred_labels, average='macro'))
    # print('recall_score', recall_score(train_labels, pred_labels, average='macro'))
    # print('accuracy_score', accuracy_score(train_labels, pred_labels))
    # print('f1_score', f1_score(train_labels, pred_labels, average='macro'))
    # # print('roc_auc_score', roc_auc_score(train_labels, pred_labels, average='macro', multi_class='ovr'))

    # print('confusion_matrix\n', confusion_matrix(train_labels, pred))
    # import pdb; pdb.set_trace()



if __name__ == '__main__':
    import numpy as np
    import torch
    import torchvision 

    c, h, w = 3, 32, 32
    a = np.zeros([3, 32, 32]).astype(np.uint8)
    for i in range(h):
        for j in range(w):
            if (i + j) % 2 == 0:
                a[:, i, j] = 255

    torchvision.io.write_jpeg(torch.from_numpy(a), 'data/checkboard.jpg')
