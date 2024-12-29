import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def rf_train_gridsearch_B(train_images, train_labels, param_grid=None, n_splits=3, random_state=42):
    # use gridsearch to apply Cross Validation in RF
    # define search grid for RF: K-fold = 3
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20]    # search deepper in BloodMNIST dataset
        }

    base_model = RandomForestClassifier(random_state=random_state)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='accuracy',                     # use accuracy sorting models
        cv=skf,                 
        n_jobs=-1,                              # concurrent processing, use all CPU core
        verbose=1,                              # show search information
        refit=True                              # auto retrain using new Params after find it
    )

    # search on train dataset
    grid_search.fit(train_images, train_labels)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_model, best_params, best_score
