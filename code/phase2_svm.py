# importing libraries #
import os as _os

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_score
)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# every figure in this assignment is written to outputs/ as a PNG #
_os.makedirs("outputs", exist_ok=True)

# helper: saving the current figure into outputs/ #
def save_fig(filename, dpi=150):
    # filenames are descriptive & unique per script, they are always written into
    # outputs/ (relative to the project root) so re-running overwrites cleanly
    _os.makedirs("outputs", exist_ok=True)

    path = _os.path.join("outputs", filename)

    # 150 dpi keeps the plots sharp enough for the report #
    plt.savefig(path, dpi=dpi, bbox_inches='tight')

    print(f"saved figure: {path}")

    # the figure is still open, so the plt.show() which follows can display it #

# setting random seed for reproducibility #
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# loading windowed datasets #
# windowed feature datasets live in data/ (paths are relative to the project root) #
def data_path(filename):
    preferred = _os.path.join("data", filename)
    if _os.path.exists(preferred):
        return preferred
    if _os.path.exists(filename):
        print(f"note: {filename} is still in the project root; "
              f"using it. move it into data/ to match the documented layout.")
        return filename
    raise FileNotFoundError(
        f"{filename} not found in data/ or in the project root. "
        "Run code/phase1.py first (from the project root) to generate it."
    )

features_30 = pd.read_csv(data_path("features_30.csv"))
features_60 = pd.read_csv(data_path("features_60.csv"))
features_120 = pd.read_csv(data_path("features_120.csv"))

# Data Preparation Before Start of Phase 2 SVM #
# preparing window size 30 #

# separating input features (X) from target label (y) #
X_30 = features_30.drop('label', axis=1)
y_30 = features_30['label']

# splitting data into training & test sets #
# using a common 80/20 split #
# using 42 as fixed random seed #
# stratify keeps approximately same proportion of acid (1) & non acid (0) samples in both training & test sets #
X_train_30, X_test_30, y_train_30, y_test_30 = train_test_split(
    X_30,
    y_30,
    test_size=0.2,
    random_state=42,
    stratify=y_30
)

# scaling #
scaler_30 = StandardScaler()
X_train_30_scaled = scaler_30.fit_transform(X_train_30)
X_test_30_scaled = scaler_30.transform(X_test_30)

# window size 60 #

# separating input features (X) from target label (y) #
X_60 = features_60.drop('label', axis=1)
y_60 = features_60['label']

# splitting data into training & test sets #
X_train_60, X_test_60, y_train_60, y_test_60 = train_test_split(
    X_60,
    y_60,
    test_size=0.2,
    random_state=42,
    stratify=y_60
)

# scaling #
scaler_60 = StandardScaler()
X_train_60_scaled = scaler_60.fit_transform(X_train_60)
X_test_60_scaled = scaler_60.transform(X_test_60)

# window size 120 #

# separating input features (X) from target label (y) #
X_120 = features_120.drop('label', axis=1)
y_120 = features_120['label']

# splitting data into training & test sets #
X_train_120, X_test_120, y_train_120, y_test_120 = train_test_split(
    X_120,
    y_120,
    test_size=0.2,
    random_state=42,
    stratify=y_120
)

# scaling #
scaler_120 = StandardScaler()
X_train_120_scaled = scaler_120.fit_transform(X_train_120)
X_test_120_scaled = scaler_120.transform(X_test_120)

# 5 fold cross validation setup #
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# 3.3.2: Support Vector Machines (SVM) #

# SVM 1: linear kernel #
# creating an SVM that tries to find a linear decision boundary #
svm_linear = SVC(
    kernel='linear',
    C=1.0
)

# 5 fold cross validation #
linear_scores = cross_val_score(
    svm_linear,
    X_train_30_scaled,
    y_train_30,
    cv=skf,
    scoring='accuracy'
)

print("\nSVM Linear Kernel - 5-Fold Cross-Validation Results")

for i, score in enumerate(linear_scores, start=1):
    print(f"Fold {i} Validation Accuracy: {score:.4f}")

print(f"Mean Validation Accuracy: {np.mean(linear_scores):.4f}")
print(f"Standard Deviation: {np.std(linear_scores):.4f}")

# SVM 2: polynomial kernel #
# testing different polynomial degrees #

poly_degrees = [2, 3, 4]
poly_scores = []

for degree in poly_degrees:

    svm_poly = SVC(
        kernel='poly',
        C=1.0,
        degree=degree,
        gamma='scale'
    )

    scores = cross_val_score(
        svm_poly,
        X_train_30_scaled,
        y_train_30,
        cv=skf,
        scoring='accuracy'
    )

    mean_accuracy = scores.mean()
    std_accuracy = scores.std()

    poly_scores.append(mean_accuracy)

    print(f"Polynomial degree = {degree}")

    for fold, accuracy in enumerate(scores, start=1):
        print(f"Fold {fold}: {accuracy:.4f}")

    print(f"Mean CV accuracy: {mean_accuracy:.4f}")
    print(f"Standard deviation: {std_accuracy:.4f}")
    print()

# SVM 3: RBF kernel #
# tuning hyperparameters C & gamma with GridSearchCV #
C_values = [0.1, 1, 10, 100]
gamma_values = [0.001, 0.01, 0.1, 1]

# the search space that GridSearchCV walks through #
param_grid = {
    'C': C_values,
    'gamma': gamma_values
}

# every C & gamma pair is scored by 5 fold cross validation #
# refit=True (the default) also fits the best estimator on the whole training set #
grid_search = GridSearchCV(
    estimator=SVC(kernel='rbf'),
    param_grid=param_grid,
    scoring='accuracy',
    cv=skf,
    refit=True,
    n_jobs=-1
)

print("\nRunning GridSearchCV over the RBF C & gamma grid "
      f"({len(C_values) * len(gamma_values)} candidates x {skf.get_n_splits()} folds) ...")

# fitting the grid search runs the whole cross validated experiment #
grid_search.fit(X_train_30_scaled, y_train_30)

cv_results = grid_search.cv_results_

# mean cross validation accuracy of every C & gamma pair #
mean_scores = cv_results['mean_test_score']
std_scores = cv_results['std_test_score']

for params, mean_score, std_score in zip(cv_results['params'], mean_scores, std_scores):
    print(
        f"C={params['C']}, gamma={params['gamma']}, "
        f"Mean CV Accuracy={mean_score:.4f} (+/- {std_score:.4f})"
    )

# the heatmap is built straight out of cv_results_ #
# GridSearchCV reports one row per C & gamma pair, in the order the grid was defined #
# (C slowest, gamma fastest), so an explicit position map is used to place each row
# at its (C, gamma) cell instead of relying on that ordering #
param_index_map = {}

for position, params in enumerate(cv_results['params']):
    row = C_values.index(params['C'])
    column = gamma_values.index(params['gamma'])
    param_index_map[(row, column)] = position

# filled with NaN so a grid cell that was never scored would show up as a blank
# heatmap cell instead of a silent 0.0 accuracy #
rbf_results = np.full((len(C_values), len(gamma_values)), np.nan)

for (row, column), position in param_index_map.items():
    rbf_results[row, column] = mean_scores[position]

assert not np.isnan(rbf_results).any(), "GridSearchCV did not report a score for every C & gamma pair"

# creating RBF C & gamma heatmaps #
plt.figure(figsize=(8, 6))

sns.heatmap(
    rbf_results,
    annot=True,
    fmt=".4f",
    xticklabels=gamma_values,
    yticklabels=C_values
)

plt.title("Cross-Validation Accuracy With Respect to C & Gamma")
plt.xlabel("Gamma")
plt.ylabel("C")
plt.tight_layout()
save_fig("phase2_svm_rbf_c_gamma_heatmap.png")
plt.show()

# identifying best RBF hyperparameters #
# GridSearchCV picked the highest mean cross validation accuracy of the grid #
best_C = grid_search.best_params_['C']
best_gamma = grid_search.best_params_['gamma']
best_rbf_accuracy = grid_search.best_score_

# the best cell & the heatmap are both taken from the same cv_results_, so they must
# agree; if GridSearchCV ever reported a different cell, the script fails here instead
# of printing a best C & gamma that the figure contradicts #
best_row = C_values.index(best_C)
best_column = gamma_values.index(best_gamma)

assert rbf_results[best_row, best_column] == best_rbf_accuracy, (
    "best_params_ does not point at the highest cell of the heatmap"
)

assert np.isclose(best_rbf_accuracy, rbf_results.max()), (
    "best_score_ is not the maximum of the heatmap built from cv_results_"
)

print("\nC & Gamma Value Giving Highest Accuracy for SVM RBF")
print(f"Best C: {best_C}")
print(f"Best Gamma: {best_gamma}")
print(f"Best Mean CV Accuracy: {best_rbf_accuracy:.4f}")

# comparing effect of window size on SVM performance #
# using RBF SVM instead of linear SVM due to greater accuracy #
# taking the best RBF SVM hyperparameters found by GridSearchCV above #
# C & gamma are constant, only window size changes #

# the training data & labels of every window size, so the same experiment
# can be repeated once per window size without duplicating the code #
window_sets = {
    30: (X_train_30_scaled, y_train_30),
    60: (X_train_60_scaled, y_train_60),
    120: (X_train_120_scaled, y_train_120)
}

for window_size, (X_window_scaled, y_window) in window_sets.items():

    svm_rbf_window = SVC(
        kernel='rbf',
        C=best_C,
        gamma=best_gamma
    )

    scores_window = cross_val_score(
        svm_rbf_window,
        X_window_scaled,
        y_window,
        cv=skf,
        scoring='accuracy'
    )

    print(f"\nRBF SVM - Window Size {window_size}")

    for i, score in enumerate(scores_window, start=1):
        print(f"Fold {i} Validation Accuracy: {score:.4f}")

    print(f"Mean Validation Accuracy: {np.mean(scores_window):.4f}")
    print(f"Standard Deviation: {np.std(scores_window):.4f}")
