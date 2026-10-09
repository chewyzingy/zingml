# importing libraries #
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# setting random seed for reproducibility #
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# loading windowed datasets #
features_30 = pd.read_csv("features_30.csv")
features_60 = pd.read_csv("features_60.csv")
features_120 = pd.read_csv("features_120.csv")

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
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

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

# print("\nSVM Linear Kernel - 5-Fold Cross-Validation Results")

# for i, score in enumerate(linear_scores, start=1):
    # print(f"Fold {i} Validation Accuracy: {score:.4f}")

# print(f"Mean Validation Accuracy: {np.mean(linear_scores):.4f}")
# print(f"Standard Deviation: {np.std(linear_scores):.4f}")

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
# tuning hyperparameters C & gamma #
C_values = [0.1, 1, 10, 100]
gamma_values = [0.001, 0.01, 0.1, 1]

rbf_results = np.zeros((len(C_values), len(gamma_values)))

for i, C in enumerate(C_values):
    for j, gamma in enumerate(gamma_values):

        svm_rbf = SVC(
            kernel='rbf',
            C=C,
            gamma=gamma
        )
        # 5 fold cross validation #
        scores = cross_val_score(
            svm_rbf,
            X_train_30_scaled,
            y_train_30,
            cv=skf,
            scoring='accuracy'
        )

        rbf_results[i, j] = np.mean(scores)

        print(
            f"C={C}, gamma={gamma}, "
            f"Mean CV Accuracy={np.mean(scores):.4f}"
        )

# creating RBF C & gamma heatmaps #
# plt.figure(figsize=(8, 6))

# sns.heatmap(
    # rbf_results,
    # annot=True,
    # fmt=".4f",
    # xticklabels=gamma_values,
    # yticklabels=C_values
# )

# plt.title("Cross-Validation Accuracy With Respect to C & Gamma")
# plt.xlabel("Gamma")
# plt.ylabel("C")
# plt.tight_layout()
# plt.show()

# identifying best RBF hyperparameters #
best_position = np.unravel_index(

    # finding highest accuracy #
    np.argmax(rbf_results),
    rbf_results.shape
)

# stating which C & gamma produced highest accuracy #
best_C = C_values[best_position[0]]
best_gamma = gamma_values[best_position[1]]
best_rbf_accuracy = rbf_results[best_position]

# print("\nC & Gamma Value Giving Highest Accuracy for SVM RBF")
# print(f"Best C: {best_C}")
# print(f"Best Gamma: {best_gamma}")
# print(f"Best Mean CV Accuracy: {best_rbf_accuracy:.4f}")

# comparing effect of window size on SVM performance #
# using RBF SVM instead of linear SVM due to greater accuracy #
# taking the best RBF SVM hyperparameters found from tuning #
# C & gamma are constant, only window size changes #

# for window size 30 #
svm_rbf_30 = SVC(
    kernel='rbf',
    C=100,
    gamma=0.01
)

scores_30 = cross_val_score(
    svm_rbf_30,
    X_train_30_scaled,
    y_train_30,
    cv=skf,
    scoring='accuracy'
)

# print("\nRBF SVM - Window Size 30")

# for i, score in enumerate(scores_30, start=1):
    # print(f"Fold {i} Validation Accuracy: {score:.4f}")

# print(f"Mean Validation Accuracy: {np.mean(scores_30):.4f}")
# print(f"Standard Deviation: {np.std(scores_30):.4f}")

# for window size 60 #
svm_rbf_60 = SVC(
    kernel='rbf',
    C=100,
    gamma=0.01
)

scores_60 = cross_val_score(
    svm_rbf_60,
    X_train_60_scaled,
    y_train_60,
    cv=skf,
    scoring='accuracy'
)

# print("\nRBF SVM - Window Size 60")

# for i, score in enumerate(scores_60, start=1):
    # print(f"Fold {i} Validation Accuracy: {score:.4f}")

# print(f"Mean Validation Accuracy: {np.mean(scores_60):.4f}")
# print(f"Standard Deviation: {np.std(scores_60):.4f}")

# for window size 120 #
svm_rbf_120 = SVC(
    kernel='rbf',
    C=100,
    gamma=0.01
)

scores_120 = cross_val_score(
    svm_rbf_120,
    X_train_120_scaled,
    y_train_120,
    cv=skf,
    scoring='accuracy'
)

# print("\nRBF SVM - Window Size 120")

# for i, score in enumerate(scores_120, start=1):
    # print(f"Fold {i} Validation Accuracy: {score:.4f}")

# print(f"Mean Validation Accuracy: {np.mean(scores_120):.4f}")
# print(f"Standard Deviation: {np.std(scores_120):.4f}")