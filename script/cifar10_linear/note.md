pretrained_feature_extractor: path to the pretrained encoder

poison_data: basename of the poison information file (cifar10) or link folder (ImageNet100). It should be like `cifar10-resnet18-con-simclr-43-0.500-0-0.9531`, we need this just for loading the label of the anchor point to calculate the ASR.