import numpy as np
import torch
from torchvision import transforms
from collections import Counter

def load_bloodmnist(npz_path='Datasets/BloodMNIST.npz'):
    # load blood data
    data = np.load(npz_path)

    train_images = data['train_images']     # shape: 11959, 28, 28
    val_images   = data['val_images']       # shape: 1715, 28, 28
    test_images  = data['test_images']      # shape: 3421, 28, 28

    train_labels = data['train_labels'].reshape(-1)  # shape: 11959, 1, 1
    val_labels   = data['val_labels'].reshape(-1)    # shape: 1715, 1, 1
    test_labels  = data['test_labels'].reshape(-1)   # shape: 3421, 1, 1

    # # count different numbers of samples in each classes:
    # label_counts = Counter(train_labels)
    # for label in range(8):
    #     print(f"Class {label}: {label_counts[label]} samples")
    # # Class 0: 852 samples
    # # Class 1: 2181 samples
    # # Class 2: 1085 samples
    # # Class 3: 2026 samples
    # # Class 4: 849 samples
    # # Class 5: 993 samples
    # # Class 6: 2330 samples
    # # Class 7: 1643 samples

    return train_images, val_images, test_images, train_labels, val_labels, test_labels

def preprocess_RF_B(images):
    # flat [N,28,28] to [N,784] then normalize to [0,1]
    N = images.shape[0]
    images_flat = images.reshape(N, -1).astype(np.float32) / 255.0
    return images_flat

def preprocess_ResNet_B(images):
    images_float = images.astype(np.float32) / 255.0
    mean = images_float.mean()
    std = images_float.std()

    transform = transforms.Compose([
        transforms.ToTensor(),               # -> [1, H, W]
        transforms.Normalize(mean=[mean], std=[std])  # normalize signal channel
    ])
    
    # get tenser with shape [N, 1, 28, 28]
    single_channel_tensor = torch.stack([transform(img) for img in images_float])
    # copy 1 channel into 3 channels（C=3, H=28, W=28）to fit 3 channel input on ResNet
    three_channel_tensor = single_channel_tensor.repeat(1, 3, 1, 1)
    return single_channel_tensor, mean, std