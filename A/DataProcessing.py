import numpy as np

def load_breastmnist(npz_path = 'Datasets/BreastMNIST.npz'):
    data = np.load(npz_path)
    # print(data.files) # ['train_images', 'val_images', 'test_images', 'train_labels', 'val_labels', 'test_labels']
    train_images = data['train_images']     # shape: 546, 28, 28
    val_images = data['val_images']         # shape: 78, 28, 28
    test_images = data['test_images']       # shape: 156, 28, 28
    train_labels = data['train_labels']     # shape: 546, 1, 1
    val_labels = data['val_labels']         # shape: 78, 1, 1
    test_labels = data['test_labels']       # shape: 156, 1, 1
    return train_images, val_images, test_images, train_labels, val_labels, test_labels

# test:
train_images, val_images, test_images, train_labels, val_labels, test_labels = load_breastmnist()
print(len(train_images[0][0]))
print(len(val_images[0][0]))
print(len(test_images[0][0]))
print(len(train_labels[0]))
print(len(val_labels[0]))
print(len(test_labels[0]))