import numpy as np
import torch
from torchvision import transforms

def load_resplit_breastmnist(npz_path = 'Datasets/BreastMNIST.npz', train_size=546, val_size=78, test_size=156, random_state=42):
    # load data
    data = np.load(npz_path)
    # print(data.files) # ['train_images', 'val_images', 'test_images', 'train_labels', 'val_labels', 'test_labels']
    train_images = data['train_images']     # shape: 546, 28, 28
    val_images = data['val_images']         # shape: 78, 28, 28
    test_images = data['test_images']       # shape: 156, 28, 28
    train_labels = data['train_labels']     # shape: 546, 1, 1
    val_labels = data['val_labels']         # shape: 78, 1, 1
    test_labels = data['test_labels']       # shape: 156, 1, 1
    # print("data loaded")
    
    # merge, and randomly resplit the dataset
    # merge images and labels
    all_images = np.concatenate([train_images, val_images, test_images], axis=0)
    all_labels = np.concatenate([train_labels, val_labels, test_labels], axis=0)

    # reshap and random data
    all_labels = all_labels.reshape(-1)
    np.random.seed(random_state)
    total_size = all_images.shape[0]
    indices = np.random.permutation(total_size)

    all_images = all_images[indices]
    all_labels = all_labels[indices]

    # resplit data
    end_train = train_size
    end_val = train_size + val_size

    train_images = all_images[:end_train]
    train_labels = all_labels[:end_train]

    val_images = all_images[end_train:end_val]
    val_labels = all_labels[end_train:end_val]

    test_images = all_images[end_val:]
    test_labels = all_labels[end_val:]
    # print("data resplited")

    return train_images, val_images, test_images, train_labels, val_labels, test_labels

def preprocess_SVM_RF_A(images):
    # flat [N,28,28] to [N,784] then normalize to [0,1]
    N = images.shape[0]
    images_flat = images.reshape(N, -1).astype(np.float32)/255.0
    print(type(images_flat))
    return images_flat

def preprocess_CNN_A(images):
    #reshape the data to [N, 1, 28, 28] and normalize to [0,1]
    images_float = images.astype(np.float32) / 255.0
    mean = images_float.mean()
    std = images_float.std()

    # define the transforms
    transform = transforms.Compose([
        transforms.ToTensor(),  # NumPy [H, W] -> Tensor [1, H, W], then normalize to [0, 1]
        transforms.Normalize(mean=[mean], std=[std])  # Normalize to (x - mean) / std
    ])
    # use transform in each images
    CNN_tensor = torch.stack([transform(img) for img in images])
    print(type(CNN_tensor))
    print("mean:",mean, "std:",std)
    return CNN_tensor

# # test:
# train_images, val_images, test_images, train_labels, val_labels, test_labels = load_resplit_breastmnist()
# print(len(train_images))
# print(len(val_images))
# print(len(test_images))
# print(len(train_labels))
# print(len(val_labels))
# print(len(test_labels))
# preprocess_SVM_RF(train_images)
# preprocess_CNN(train_images)