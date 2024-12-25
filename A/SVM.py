import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def svm_train_gridsearch_A(train_images, train_labels, C_values=[0.01, 0.1, 1, 10], n_splits=3, random_state=42):
# use GridSearchCV make Cross-validation and gridsearch for SVM, to get the best C (regularization param)
# return best model, C and acc
    # define the parameter grad
    param_grid = {'C': C_values} # regularization parameter

    # initialize base SVC (support Vector Classifier)
    base_model = SVC(kernel='rbf', probability=True, random_state=random_state) # rbf: radial basis function kernel
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='accuracy',         # use accuracy sorting models
        cv=skf,
        n_jobs=-1,                  # concurrent processing, use all CPU core
        verbose=1                   # show search information
    )

    # train the model use train_images
    grid_search.fit(train_images, train_labels)

    best_model = grid_search.best_estimator_
    best_C = grid_search.best_params_['C']
    best_acc = grid_search.best_score_          # average CV acc

    return best_model, best_C, best_acc