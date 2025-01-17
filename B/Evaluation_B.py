import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics_B(y_true, y_pred):
    # calculate Accuracy, Precision, Recall, F1 score for multiclass
    # average = macro -> multi classification
    acc = accuracy_score(y_true, y_pred)

    # macro: Calculate the metrics for each category and then get average
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, prec, rec, f1

def plot_metrics_over_epochs_B(metrics_dict, model_name='TaskB-ResNet34'):
    # Plotting the variation of accuracy, precision, recall, f1 with epoch
    # during training & validation in multi-classification scenarios.

    # metrics_dict：(example show)
    # {
    #     'train_acc': [], 'train_prec': [], 'train_rec': [], 'train_f1': [],
    #     'val_acc':   [], 'val_prec':   [], 'val_rec':   [], 'val_f1':   []
    # }
    
    epochs = range(1, len(metrics_dict['train_acc']) + 1)

    # plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics_dict['train_loss'], label='Train Loss', color='blue')
    plt.plot(epochs, metrics_dict['val_loss'], label='Val Loss', color='orange')
    plt.title(f"{model_name} - Learning Curve", fontsize=16)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"{model_name} Task A Training Metrics", fontsize=16)
    # Accuracy
    axs[0, 0].plot(epochs, metrics_dict['train_acc'], label='Train Accuracy')
    axs[0, 0].plot(epochs, metrics_dict['val_acc'], label='Val Accuracy')
    axs[0, 0].set_title(f'{model_name} - Accuracy')
    axs[0, 0].legend()

    # Precision
    axs[0, 1].plot(epochs, metrics_dict['train_prec'], label='Train Precision')
    axs[0, 1].plot(epochs, metrics_dict['val_prec'], label='Val Precision')
    axs[0, 1].set_title(f'{model_name} - Precision')
    axs[0, 1].legend()

    # Recall
    axs[1, 0].plot(epochs, metrics_dict['train_rec'], label='Train Recall')
    axs[1, 0].plot(epochs, metrics_dict['val_rec'], label='Val Recall')
    axs[1, 0].set_title(f'{model_name} - Recall')
    axs[1, 0].legend()

    # F1
    axs[1, 1].plot(epochs, metrics_dict['train_f1'], label='Train F1')
    axs[1, 1].plot(epochs, metrics_dict['val_f1'], label='Val F1')
    axs[1, 1].set_title(f'{model_name} - F1')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()
