import numpy as np

from A.DataProcessing import load_resplit_breastmnist, preprocess_SVM_RF, preprocess_CNN
from A.SVM import svm_train_gridsearch


def main():

    train_images, val_images, test_images, train_labels, val_labels, test_labels = load_resplit_breastmnist()

    train_images_svm = preprocess_SVM_RF(train_images) 
    
    svm_model, best_C, svm_cv_acc = svm_train_gridsearch(train_images_svm, train_labels)

    print("Best C:",best_C, "SVM_acc:", svm_cv_acc)



if __name__ == "__main__":
    main()
