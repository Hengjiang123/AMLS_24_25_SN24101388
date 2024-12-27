import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

from A.DataProcessing import load_resplit_breastmnist, preprocess_SVM_RF_A, preprocess_CNN_A
from A.Evaluation import calculate_metrics, plot_metrics_over_epochs
from A.SVM import svm_train_gridsearch_A
from A.RandomForest import rf_train_gridsearch_A
from A.CNN import CNN_A, dataloaders_A, CNN_train_A

def main():
    ############### Task A ################
    print("\n############### Task A ################\n")

    # load and randomly split data as Train Val and Test dataset
    train_images, val_images, test_images, train_labels, val_labels, test_labels = load_resplit_breastmnist()
    
    print("Data Shapes:")
    print("Train:", train_images.shape, train_labels.shape)
    print("Val:", val_images.shape, val_labels.shape)
    print("Test:", test_images.shape, test_labels.shape)

    # Preprocess, plant and normalization 
    train_images_svm_rf = preprocess_SVM_RF_A(train_images) 
    val_images_svm_rf = preprocess_SVM_RF_A(val_images)
    test_images_svm_rf = preprocess_SVM_RF_A(test_images)

    ##### SVM Training #####
    print("\n=== SVM Training ===")

    # make cross validation on train dataset to get best C, then use C train the model
    svm_model, svm_best_C, svm_cv_acc = svm_train_gridsearch_A(train_images_svm_rf, train_labels)
    print(f"Best C = {svm_best_C}, SVM_cv_acc = {svm_cv_acc:.4f}")

    # Evaluate on val dataset, and calculate 4 matrics
    val_labels_pred_svm = svm_model.predict(val_images_svm_rf)
    val_acc_svm, val_prec_svm, val_rec_svm, val_f1_svm = calculate_metrics(val_labels, val_labels_pred_svm)
    print(f"SVM Val dataset: Accuracy={val_acc_svm:.4f}, Precision={val_prec_svm:.4f}, Recall={val_rec_svm:.4f}, F1={val_f1_svm:.4f}") 
    
    # Ecaluate on test dataset, and calculate 4 matrics
    test_labels_pred_svm = svm_model.predict(test_images_svm_rf)
    test_acc_svm, test_prec_svm, test_rec_svm, test_f1_svm = calculate_metrics(test_labels, test_labels_pred_svm)
    print(f"SVM Test dataset: Accuracy={test_acc_svm:.4f}, Precision={test_prec_svm:.4f}, Recall={test_rec_svm:.4f}, F1={test_f1_svm:.4f}") 
   

    ##### Random Forest Training #####
    print("\n=== Random Forest Training ===")

    # train Random Forest model with cross validation, and get best model and params
    rf_model, rf_best_params, rf_cv_acc = rf_train_gridsearch_A(train_images_svm_rf, train_labels)
    print(f"RF best params={rf_best_params}, RF_cv_acc={rf_cv_acc:.4f}")

    # Evaluate on val dataset, and calculate 4 matrics
    val_labels_pred_rf = rf_model.predict(val_images_svm_rf)
    val_acc_rf, val_prec_rf, val_rec_rf, val_f1_rf = calculate_metrics(val_labels, val_labels_pred_rf)
    print(f"RF Val dataset: Accuracy={val_acc_rf:.4f}, Precision={val_prec_rf:.4f}, Recall={val_rec_rf:.4f}, F1={val_f1_rf:.4f}")

    # Evaluate on test dataset, and calculate 4 matrics
    test_labels_pred_rf = rf_model.predict(test_images_svm_rf)
    test_acc_rf, test_prec_rf, test_rec_rf, test_f1_rf = calculate_metrics(test_labels, test_labels_pred_rf)
    print(f"RF Test dataset: Accuracy={test_acc_rf:.4f}, Precision={test_prec_rf:.4f}, Recall={test_rec_rf:.4f}, F1={test_f1_rf:.4f}")


    ##### CNN Training #####  
    print("\n=== CNN Training ===")

    # preprocess data for CNN
    train_images_cnn = preprocess_CNN_A(train_images)
    val_images_cnn = preprocess_CNN_A(val_images)
    test_images_cnn = preprocess_CNN_A(test_images)

    # create Dataloader
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader = dataloaders_A(train_images_cnn, train_labels, val_images_cnn, val_labels, batch_size=32)

    # initialize CNN model and train
    cnn_model = CNN_A(num_classes=2)
    cnn_model, metrics_record_cnn = CNN_train_A(cnn_model, train_loader, val_loader, epochs=10, lr=1e-3, weight_decay=1e-4, device='cpu')

    # plot recorded metrics when training CNN
    plot_metrics_over_epochs(metrics_record_cnn, model_name='TaskA-Light-CNN')

    # package CNN test dataset
    test_dataset = TensorDataset(
        torch.tensor(test_images_cnn, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long)
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    cnn_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = cnn_model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    test_acc_cnn, test_prec_cnn, test_rec_cnn, test_f1_cnn = calculate_metrics(all_labels, all_preds)
    print(f"CNN Test dataset: Accuracy={test_acc_cnn:.4f}, Precision={test_prec_cnn:.4f}, Recall={test_rec_cnn:.4f}, F1={test_f1_cnn:.4f}")

    # here reoutput metrics of three models in test data and compare their performance
    print("\n=== TaskA: Final Comparison on Test Set ===")
    print("Model\t\tAcc\tPrec\tRec\tF1")
    print(f"SVM\t\t{test_acc_svm:.4f}\t{test_prec_svm:.4f}\t{test_rec_svm:.4f}\t{test_f1_svm:.4f}")
    print(f"RF\t\t{test_acc_rf:.4f}\t{test_prec_rf:.4f}\t{test_rec_rf:.4f}\t{test_f1_rf:.4f}")
    print(f"CNN\t\t{test_acc_cnn:.4f}\t{test_prec_cnn:.4f}\t{test_rec_cnn:.4f}\t{test_f1_cnn:.4f}")



    ############### Task B ################

if __name__ == "__main__":
    main()
