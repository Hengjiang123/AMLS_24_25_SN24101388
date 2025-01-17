import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import ResNet34_Weights
from collections import Counter

from A.DataProcessing_A import load_resplit_breastmnist, preprocess_SVM_RF_A, preprocess_CNN_A
from A.Evaluation_A import calculate_metrics_A, plot_metrics_over_epochs_A
from A.SVM_A import svm_train_gridsearch_A
from A.RandomForest_A import rf_train_gridsearch_A
from A.CNN_A import CNN_A, dataloaders_A, CNN_train_A

from B.DataProcessing_B import load_bloodmnist, preprocess_RF_B, preprocess_ResNet_B
from B.Evaluation_B import calculate_metrics_B, plot_metrics_over_epochs_B
from B.RandomForest_B import rf_train_gridsearch_B
from B.ResNet_B import ResNet34_B, ResNet_train_B

def TaskA():
    ############### Task A ################
    print("\n############### Task A ################\n")

    # load and randomly split data as Train Val and Test dataset
    train_images, val_images, test_images, train_labels, val_labels, test_labels = load_resplit_breastmnist()
    
    print("Data Shapes:")
    print("Train:", train_images.shape, train_labels.shape)
    print("Val:", val_images.shape, val_labels.shape)
    print("Test:", test_images.shape, test_labels.shape)

    print("\nTrain labels distribution:", Counter(train_labels))
    print("Val labels distribution:", Counter(val_labels))
    print("Test labels distribution:", Counter(test_labels))

    # Preprocess, plant and normalization 
    train_images_svm_rf = preprocess_SVM_RF_A(train_images) 
    val_images_svm_rf = preprocess_SVM_RF_A(val_images)
    test_images_svm_rf = preprocess_SVM_RF_A(test_images)

    ##### SVM Training #####
    print("\n=== SVM Training ===")

    # Make cross validation on train dataset to get best C, then use C train the model
    svm_model, svm_best_C, svm_cv_acc = svm_train_gridsearch_A(train_images_svm_rf, train_labels)
    print(f"Best C = {svm_best_C}, SVM_cv_acc = {svm_cv_acc:.4f}")

    # Evaluate on TRAIN dataset
    train_labels_pred_svm = svm_model.predict(train_images_svm_rf)
    train_acc_svm, train_prec_svm, train_rec_svm, train_f1_svm = calculate_metrics_A(train_labels, train_labels_pred_svm)
    print(f"SVM Train dataset: Accuracy={train_acc_svm:.4f}, Precision={train_prec_svm:.4f}, Recall={train_rec_svm:.4f}, F1={train_f1_svm:.4f}") 

    # Evaluate on val dataset, and calculate 4 matrics
    val_labels_pred_svm = svm_model.predict(val_images_svm_rf)
    val_acc_svm, val_prec_svm, val_rec_svm, val_f1_svm = calculate_metrics_A(val_labels, val_labels_pred_svm)
    print(f"SVM Val dataset: Accuracy={val_acc_svm:.4f}, Precision={val_prec_svm:.4f}, Recall={val_rec_svm:.4f}, F1={val_f1_svm:.4f}") 
    
    # Ecaluate on test dataset, and calculate 4 matrics
    test_labels_pred_svm = svm_model.predict(test_images_svm_rf)
    test_acc_svm, test_prec_svm, test_rec_svm, test_f1_svm = calculate_metrics_A(test_labels, test_labels_pred_svm)
    print(f"SVM Test dataset: Accuracy={test_acc_svm:.4f}, Precision={test_prec_svm:.4f}, Recall={test_rec_svm:.4f}, F1={test_f1_svm:.4f}") 
   

    ##### Random Forest Training #####
    print("\n=== RandomForest Training ===")

    # train Random Forest model with cross validation, and get best model and params
    rf_model, rf_best_params, rf_cv_acc = rf_train_gridsearch_A(train_images_svm_rf, train_labels)
    print(f"RF best params={rf_best_params}, RF_cv_acc={rf_cv_acc:.4f}")
    
    # Evaluate on TRAIN dataset
    train_labels_pred_rf = rf_model.predict(train_images_svm_rf)
    train_acc_rf, train_prec_rf, train_rec_rf, train_f1_rf = calculate_metrics_A(train_labels, train_labels_pred_rf)
    print(f"RF Train dataset: Accuracy={train_acc_rf:.4f}, Precision={train_prec_rf:.4f}, Recall={train_rec_rf:.4f}, F1={train_f1_rf:.4f}")

    # Evaluate on val dataset, and calculate 4 matrics
    val_labels_pred_rf = rf_model.predict(val_images_svm_rf)
    val_acc_rf, val_prec_rf, val_rec_rf, val_f1_rf = calculate_metrics_A(val_labels, val_labels_pred_rf)
    print(f"RF Val dataset: Accuracy={val_acc_rf:.4f}, Precision={val_prec_rf:.4f}, Recall={val_rec_rf:.4f}, F1={val_f1_rf:.4f}")

    # Evaluate on test dataset, and calculate 4 matrics
    test_labels_pred_rf = rf_model.predict(test_images_svm_rf)
    test_acc_rf, test_prec_rf, test_rec_rf, test_f1_rf = calculate_metrics_A(test_labels, test_labels_pred_rf)
    print(f"RF Test dataset: Accuracy={test_acc_rf:.4f}, Precision={test_prec_rf:.4f}, Recall={test_rec_rf:.4f}, F1={test_f1_rf:.4f}")


    ##### CNN Training #####  
    print("\n=== CNN Training ===")

    # preprocess data for CNN
    train_images_cnn = preprocess_CNN_A(train_images)
    val_images_cnn = preprocess_CNN_A(val_images)
    test_images_cnn = preprocess_CNN_A(test_images)
    # print("test_images_cnn:", type(test_images_cnn), test_images_cnn.shape)
    # print("test_labels:", type(test_labels), test_labels.shape)

    # create Dataloader
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader = dataloaders_A(train_images_cnn, train_labels, val_images_cnn, val_labels, batch_size=32)

    # initialize CNN model and train
    cnn_model = CNN_A(num_classes=2)
    cnn_model, metrics_record_cnn = CNN_train_A(cnn_model, train_loader, val_loader, epochs=100, lr=1e-3, weight_decay=1e-4, device=device)

    # plot recorded metrics when training CNN
    plot_metrics_over_epochs_A(metrics_record_cnn, model_name='TaskA-Light-CNN')

    # package CNN test dataset
    test_dataset = TensorDataset(
        test_images_cnn.clone().detach().to(torch.float32),
        torch.tensor(test_labels, dtype=torch.long)
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # evaluate on test dataset
    cnn_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = cnn_model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())

    test_acc_cnn, test_prec_cnn, test_rec_cnn, test_f1_cnn = calculate_metrics_A(all_labels, all_preds)
    print(f"CNN Test dataset: Accuracy={test_acc_cnn:.4f}, Precision={test_prec_cnn:.4f}, Recall={test_rec_cnn:.4f}, F1={test_f1_cnn:.4f}")

    # here reoutput metrics of three models in test data and compare their performance
    print("\n=== TaskA: Final Comparison on Test Set ===")
    print("Model\t\tAcc\tPrec\tRec\tF1")
    print(f"SVM\t\t{test_acc_svm:.4f}\t{test_prec_svm:.4f}\t{test_rec_svm:.4f}\t{test_f1_svm:.4f}")
    print(f"RF\t\t{test_acc_rf:.4f}\t{test_prec_rf:.4f}\t{test_rec_rf:.4f}\t{test_f1_rf:.4f}")
    print(f"CNN\t\t{test_acc_cnn:.4f}\t{test_prec_cnn:.4f}\t{test_rec_cnn:.4f}\t{test_f1_cnn:.4f}")


def TaskB():
    ############### Task B ################
    print("\n############### Task B ################\n")

    # load BloodMNIST dataset
    train_images, val_images, test_images, train_labels, val_labels, test_labels = load_bloodmnist()
    print("Data Shapes:")
    print("Train:", train_images.shape, train_labels.shape)
    print("Val:", val_images.shape, val_labels.shape)
    print("Test:", test_images.shape, test_labels.shape)

    print("\nTrain labels distribution:", Counter(train_labels))
    print("Val labels distribution:", Counter(val_labels))
    print("Test labels distribution:", Counter(test_labels))

    ##### RandomForest Training #####
    print("\n=== RandomForest Training ===")

    # Preprocess, plant and normalization 
    train_images_rf = preprocess_RF_B(train_images)
    val_images_rf   = preprocess_RF_B(val_images)
    test_images_rf  = preprocess_RF_B(test_images)

    # Make cross validation on train dataset to get best params, then use it train the model
    rf_model, rf_best_params, rf_cv_acc = rf_train_gridsearch_B(train_images_rf, train_labels)
    print(f"RF best params={rf_best_params}, RF_cv_acc={rf_cv_acc:.4f}")

    # Evaluate on TRAIN dataset
    train_rf_pred = rf_model.predict(train_images_rf)
    train_acc_rf, train_prec_rf, train_rec_rf, train_f1_rf = calculate_metrics_B(train_labels, train_rf_pred)
    print(f"RF Train dataset: Accuracy={train_acc_rf:.4f}, Precision={train_prec_rf:.4f}, Recall={train_rec_rf:.4f}, F1={train_f1_rf:.4f}")

    # Evaluate on val dataset, and calculate 4 matrics
    val_rf_pred = rf_model.predict(val_images_rf)
    val_acc_rf, val_prec_rf, val_rec_rf, val_f1_rf = calculate_metrics_B(val_labels, val_rf_pred)
    print(f"RF Val: Accuracy={val_acc_rf:.4f}, Precision={val_prec_rf:.4f}, Recall={val_rec_rf:.4f}, F1={val_f1_rf:.4f}")

    # Evaluate on test dataset, and calculate 4 matrics
    test_rf_pred = rf_model.predict(test_images_rf)
    test_acc_rf, test_prec_rf, test_rec_rf, test_f1_rf = calculate_metrics_B(test_labels, test_rf_pred)
    print(f"RF Test: Accuracy={test_acc_rf:.4f}, Precision={test_prec_rf:.4f}, Recall={test_rec_rf:.4f}, F1={test_f1_rf:.4f}")


    ##### ResNet Training #####
    print("\n=== ResNet Training ===")

    # data preprocess, turn to 3 channels and normalization
    train_images_resnet, mean_b, std_b = preprocess_ResNet_B(train_images)
    val_images_resnet, _, _ = preprocess_ResNet_B(val_images)
    test_images_resnet, _, _ = preprocess_ResNet_B(test_images)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create Dataloader
    train_dataset = TensorDataset(train_images_resnet, torch.tensor(train_labels, dtype=torch.long))
    val_dataset   = TensorDataset(val_images_resnet, torch.tensor(val_labels, dtype=torch.long))
    test_dataset  = TensorDataset(test_images_resnet, torch.tensor(test_labels, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # initialize ResNet and training
    resnet_model = ResNet34_B(num_classes=8, weights=ResNet34_Weights.IMAGENET1K_V1)
    resnet_model, metrics_record_resnet = ResNet_train_B(
        model=resnet_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,             
        lr=1e-3,
        weight_decay=1e-4,
        device=device
    )
    
    # plot recorded metrics when training CNN
    plot_metrics_over_epochs_B(metrics_record_resnet, model_name='TaskB-ResNet34')

    # evaluate on test dataset 
    resnet_model.eval()
    all_preds = []
    all_true  = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = resnet_model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = y_batch.cpu().numpy()

            all_preds.extend(preds)
            all_true.extend(labels)

    test_acc_resnet, test_prec_resnet, test_rec_resnet, test_f1_resnet = calculate_metrics_B(all_true, all_preds)
    print(f"ResNet Test: Accuracy={test_acc_resnet:.4f}, Precision={test_prec_resnet:.4f}, Recall={test_rec_resnet:.4f}, F1={test_f1_resnet:.4f}")

    # Output metrics of two models in test data again for comparing their performance
    print("\n=== TaskB: Final Comparison on Test Set ===")
    print("Model\t\tAcc\tPrec\tRec\tF1")
    print(f"RF\t\t{test_acc_rf:.4f}\t{test_prec_rf:.4f}\t{test_rec_rf:.4f}\t{test_f1_rf:.4f}")
    print(f"ResNet34\t{test_acc_resnet:.4f}\t{test_prec_resnet:.4f}\t{test_rec_resnet:.4f}\t{test_f1_resnet:.4f}")



def main():
    # verify GPU

    # print(torch.__version__)  # check PyTorch version
    # print(torch.version.cuda)  # check support CUDA version

    # print(torch.cuda.is_available())  # return True means GPU avaliable
    # print(torch.cuda.get_device_name(0)) 

    # implement Task A and Task B 

    TaskA()
    TaskB()

if __name__ == "__main__":
    main()


