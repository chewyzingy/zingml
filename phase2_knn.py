# importing libraries #
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

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

# Data Preparation Before Start of Phase 2 kNN #
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

# 3.3.3: k Nearest Neighbours (kNN) #
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# k values to test #
# using odd k values to reduce chances of tied votes #
k_values = [1, 3, 5, 7, 9, 15, 25]

# experiment 1: euclidean distance + uniform weighting #
euclidean_uniform_scores = []

for k in k_values:

    # creating kNN model
    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric='euclidean',
        weights='uniform'
    )

    # 5-fold cross validation
    scores = cross_val_score(
        knn,
        X_train_30_scaled,
        y_train_30,
        cv=skf,
        scoring='accuracy'
    )

    # calculating mean CV accuracy for this k
    mean_accuracy = scores.mean()

    # storing result
    euclidean_uniform_scores.append(mean_accuracy)

    # displaying results
    # print(f"k = {k}")
    # print("Fold accuracies:")

    # for fold, accuracy in enumerate(scores, start=1):
        # print(f"Fold {fold}: {accuracy:.4f}")

    # print(f"Mean CV accuracy: {mean_accuracy:.4f}")
    # print()

# experiment 2: euclidean distance + distance weighting #
euclidean_distance_scores = []

for k in k_values:

    # creating kNN model
    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric='euclidean',
        weights='distance'
    )

    # 5-fold cross validation
    scores = cross_val_score(
        knn,
        X_train_30_scaled,
        y_train_30,
        cv=skf,
        scoring='accuracy'
    )

    # calculating mean CV accuracy for this k
    mean_accuracy = scores.mean()

    # storing result
    euclidean_distance_scores.append(mean_accuracy)

    # displaying results
    # print(f"k = {k}")
    # print("Fold accuracies:")

    # for fold, accuracy in enumerate(scores, start=1):
        # print(f"Fold {fold}: {accuracy:.4f}")

    # print(f"Mean CV accuracy: {mean_accuracy:.4f}")
    # print()

# experiment 3: manhattan distance + uniform weighting #
manhattan_uniform_scores = []

for k in k_values:

    # creating kNN model
    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric='manhattan',
        weights='uniform'
    )

    # 5-fold cross validation
    scores = cross_val_score(
        knn,
        X_train_30_scaled,
        y_train_30,
        cv=skf,
        scoring='accuracy'
    )

    # calculating mean CV accuracy for this k
    mean_accuracy = scores.mean()

    # storing result
    manhattan_uniform_scores.append(mean_accuracy)

    # displaying results
    # print(f"k = {k}")
    # print("Fold accuracies:")

    # for fold, accuracy in enumerate(scores, start=1):
        # print(f"Fold {fold}: {accuracy:.4f}")

    # print(f"Mean CV accuracy: {mean_accuracy:.4f}")
    # print()

# experiment 4: manhattan distance + distance weighting #
manhattan_distance_scores = []

for k in k_values:

    # creating kNN model
    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric='manhattan',
        weights='distance'
    )

    # 5-fold cross validation
    scores = cross_val_score(
        knn,
        X_train_30_scaled,
        y_train_30,
        cv=skf,
        scoring='accuracy'
    )

    # calculating mean CV accuracy for this k
    mean_accuracy = scores.mean()

    # storing result
    manhattan_distance_scores.append(mean_accuracy)

    # displaying results
    # print(f"k = {k}")
    # print("Fold accuracies:")

    # for fold, accuracy in enumerate(scores, start=1):
        # print(f"Fold {fold}: {accuracy:.4f}")

    # print(f"Mean CV accuracy: {mean_accuracy:.4f}")
    # print()

# plotting accuracy vs k for all kNN configurations #

# plt.figure(figsize=(8, 5))

# plt.plot(k_values, euclidean_uniform_scores, marker='o',
         # label='Euclidean + Uniform')

# plt.plot(k_values, euclidean_distance_scores, marker='o',
         # label='Euclidean + Distance')

# plt.plot(k_values, manhattan_uniform_scores, marker='o',
         # label='Manhattan + Uniform')

#plt.plot(k_values, manhattan_distance_scores, marker='o',
         # label='Manhattan + Distance')

# plt.xlabel('Number of Neighbours (k)')
# plt.ylabel('Mean 5-Fold CV Accuracy')
# plt.title('kNN Accuracy vs k')

# plt.xticks(k_values)
# plt.legend()
# plt.grid(True)

# plt.show()

# comparing effect of different window sizes on kNN performance #
# using manhattan + distance configuration due to greatest accuracy #
# using best k value found #
# keeping k, distance metric and weighting constant so only window size changes #

window_sizes = [30, 60, 120]

X_train_sets = [
    X_train_30_scaled,
    X_train_60_scaled,
    X_train_120_scaled
]

y_train_sets = [
    y_train_30,
    y_train_60,
    y_train_120
]

window_mean_scores = []

for window_size, X_train, y_train in zip(
    window_sizes, X_train_sets, y_train_sets
):

    knn_best = KNeighborsClassifier(
        n_neighbors=25,
        metric='manhattan',
        weights='distance'
    )

    scores = cross_val_score(
        knn_best,
        X_train,
        y_train,
        cv=skf,
        scoring='accuracy'
    )

    mean_accuracy = scores.mean()
    std_accuracy = scores.std()

    window_mean_scores.append(mean_accuracy)

    print(f"Window size = {window_size}")
    
    for fold, accuracy in enumerate(scores, start=1):
        print(f"Fold {fold}: {accuracy:.4f}")

    print(f"Mean CV accuracy: {mean_accuracy:.4f}")
    print(f"Standard deviation: {std_accuracy:.4f}")
    print()
