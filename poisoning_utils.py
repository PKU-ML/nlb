import torch
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
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

def get_poisoning_indices(anchor_feature, train_features, num_poisons):
    vals, indices = torch.topk(train_features @ anchor_feature, k=num_poisons, dim=0)
    return indices

def generate_trigger_cifar(trigger_type='checkerboard_center'):
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


def generate_trigger_imagenet(trigger_type='checkerboard_center'):
    if trigger_type == 'checkerboard_full':  # checkerboard at the center
        pattern = np.array(Image.open('./data/checkboard.jpg'))
        mask = np.ones(shape=(224, 224, 1), dtype=np.uint8)
    elif trigger_type == 'gaussian_noise':
        pattern = Image.open('./data/imagenet_gaussian_noise.jpg')
        mask = 1
    elif trigger_type == 'patch':
        pattern = Image.open('./data/trigger_10.png')
        mask = 1
    else:
        raise ValueError(
            'Please choose valid poison method: [checkerboard_1corner | checkerboard_4corner | gaussian_noise]')
    return pattern, mask

def get_trigger(dataset, trigger_type):
    if dataset in ['cifar10', 'cifar100']:
        pattern, mask = generate_trigger_cifar(trigger_type=trigger_type)
    elif dataset in ['imagenet', 'imagenet100']:
        pattern, mask = generate_trigger_imagenet(trigger_type=trigger_type)
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

def transform_dataset(dataset_name, dataset, poison_data):
    if 'cifar' in dataset_name:
        dataset.data = add_trigger(dataset.data, poison_data['pattern'], poison_data['mask'], None, poison_data['args'].trigger_alpha)
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

def split_cifar(cifar_dataset, dataset, pretrain = True):
    file_name = {
        "cifar10":{True: "cifar10_pre.txt",
            False:"cifar10_down.txt"},
        "cifar100":{True: "cifar100_pre.txt",
            False:"cifar100_down.txt"}
    }[dataset][pretrain]
    
    index = np.loadtxt("./data/" + file_name,dtype=int)
    cifar_dataset.data = cifar_dataset.data[index]
    cifar_dataset.targets = [cifar_dataset.targets[i] for i in index]
    return cifar_dataset

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
