import numpy as np

from A.DataProcessing import load_resplit_breastmnist, preprocess_SVM_RF_A, preprocess_CNN_A
from A.SVM import svm_train_gridsearch_A
from A.RandomForest import rf_train_gridsearch_A


def main():

    train_images, val_images, test_images, train_labels, val_labels, test_labels = load_resplit_breastmnist()

    train_images_svm_rf = preprocess_SVM_RF_A(train_images) 

    # test output SVM    
    svm_model, svm_best_C, svm_cv_acc = svm_train_gridsearch_A(train_images_svm_rf, train_labels)
    print("Best C:", svm_best_C, "SVM_acc:", svm_cv_acc)

    # test output rf
    rf_model, rf_best_params, rf_cv_acc = rf_train_gridsearch_A(train_images_svm_rf, train_labels)
    print("Best Params:", rf_best_params, "RF acc:", rf_cv_acc)


if __name__ == "__main__":
    main()
