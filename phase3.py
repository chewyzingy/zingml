# 3.4 Final Model Evaluation #

# importing libraries #
import setuptools.dist

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

# setting random seed for reproducibility #
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# loading selected window size dataset #
features_120 = pd.read_csv("features_120.csv")

# separating input features (X) from target label (y) #
X_120 = features_120.drop('label', axis=1)
y_120 = features_120['label']

# recreating the same 80/20 train-test split #
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

print("Training samples:", len(X_train_120))
print("Test samples:", len(X_test_120))

# final neural network model #
# using NN-4 architecture with Adam optimizer #
# using window size 120 as it achieved the highest CV accuracy #
final_nn = keras.Sequential([
    layers.Input(shape=(9,)),
    layers.Dense(
        16,
        activation='tanh',
        kernel_regularizer=keras.regularizers.l2(0.001)
    ),
    layers.Dense(
        8,
        activation='tanh',
        kernel_regularizer=keras.regularizers.l2(0.001)
    ),
    layers.Dense(
        4,
        activation='tanh',
        kernel_regularizer=keras.regularizers.l2(0.001)
    ),
    layers.Dense(1, activation='sigmoid')
])

final_nn.compile(
    optimizer=keras.optimizers.Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# training final neural network model on full training set #
final_nn.fit(
    X_train_120_scaled,
    y_train_120,
    epochs=100,
    batch_size=32,
    verbose=0
)

# evaluating final neural network on held-out test set #
nn_test_loss, nn_test_accuracy = final_nn.evaluate(
    X_test_120_scaled,
    y_test_120,
    verbose=0
)

print("\nFinal Neural Network Test Results")
print(f"Test Accuracy: {nn_test_accuracy:.4f}")
print(f"Test Loss: {nn_test_loss:.4f}")

# final SVM model #
# using RBF kernel, C = 100 & gamma = 0.01 #
# using window size 120 as it achieved the highest CV accuracy #

final_svm = SVC(
    kernel='rbf',
    C=100,
    gamma=0.01
)

# training final SVM on full training set #
final_svm.fit(
    X_train_120_scaled,
    y_train_120
)

# evaluating final SVM on held-out test set #
svm_test_accuracy = final_svm.score(
    X_test_120_scaled,
    y_test_120
)

print("\nFinal SVM Test Results")
print(f"Test Accuracy: {svm_test_accuracy:.4f}")

# final kNN model #
# using k = 25, Manhattan distance, & distance weighting #
# using window size 120 as it achieved the highest CV accuracy #

final_knn = KNeighborsClassifier(
    n_neighbors=25,
    metric='manhattan',
    weights='distance'
)

# training final k-NN on full training set #
final_knn.fit(
    X_train_120_scaled,
    y_train_120
)

# evaluating final k-NN on held-out test set #
knn_test_accuracy = final_knn.score(
    X_test_120_scaled,
    y_test_120
)

print("\nFinal k-NN Test Results")
print(f"Test Accuracy: {knn_test_accuracy:.4f}")

# generating predictions for final models #

# since neural network gives probabilities, convert them to 0 or 1 #
nn_probabilities = final_nn.predict(
    X_test_120_scaled,
    verbose=0
).ravel()

# using 0.5 as a classification threshold #
nn_predictions = (nn_probabilities >= 0.5).astype(int)

# since SVM and k-NN directly already give predicted classes #
svm_predictions = final_svm.predict(X_test_120_scaled)
knn_predictions = final_knn.predict(X_test_120_scaled)

# creating confusion matrices for all final models #
models = {
    'Neural Network': nn_predictions,
    'SVM': svm_predictions,
    'k-NN': knn_predictions
}

for model_name, predictions in models.items():

    cm = confusion_matrix(y_test_120, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Non-Acid (0)', 'Acid (1)']
    )

    display.plot()
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

# calculating final evaluation metrics (accuracy, precision, recall, f1 score) #
for model_name, predictions in models.items():

    accuracy = accuracy_score(y_test_120, predictions)
    precision = precision_score(y_test_120, predictions)
    recall = recall_score(y_test_120, predictions)
    f1 = f1_score(y_test_120, predictions)

    print(f"\n{model_name} Final Evaluation Metrics")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

# ROC curve & AUC for all final models #

# neural network probability scores #
nn_scores = nn_probabilities

# SVM decision scores #
svm_scores = final_svm.decision_function(X_test_120_scaled)

# kNN probability scores #
knn_scores = final_knn.predict_proba(X_test_120_scaled)[:, 1]

# calculating ROC curves #
nn_fpr, nn_tpr, _ = roc_curve(y_test_120, nn_scores)
svm_fpr, svm_tpr, _ = roc_curve(y_test_120, svm_scores)
knn_fpr, knn_tpr, _ = roc_curve(y_test_120, knn_scores)

# calculating AUC #
nn_auc = roc_auc_score(y_test_120, nn_scores)
svm_auc = roc_auc_score(y_test_120, svm_scores)
knn_auc = roc_auc_score(y_test_120, knn_scores)

# plotting combined ROC curve #
plt.figure()

plt.plot(nn_fpr, nn_tpr, label=f'Neural Network (AUC = {nn_auc:.3f})')
plt.plot(svm_fpr, svm_tpr, label=f'SVM (AUC = {svm_auc:.3f})')
plt.plot(knn_fpr, knn_tpr, label=f'k-NN (AUC = {knn_auc:.3f})')

# random guessing reference line #
plt.plot([0, 1], [0, 1], '--', label='Random Guessing')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Final Models')
plt.legend()
plt.grid()
plt.show()

# printing AUC values #
print("\nAUC Results")
print(f"Neural Network AUC: {nn_auc:.4f}")
print(f"SVM AUC: {svm_auc:.4f}")
print(f"k-NN AUC: {knn_auc:.4f}")

# precision-recall curves for all final models #

# calculating precision-recall curves #
nn_precision, nn_recall, _ = precision_recall_curve(
    y_test_120, nn_scores
)

svm_precision, svm_recall, _ = precision_recall_curve(
    y_test_120, svm_scores
)

knn_precision, knn_recall, _ = precision_recall_curve(
    y_test_120, knn_scores
)

# calculating average precision (AP) #
# summarises precision-recall performance into a number #
nn_ap = average_precision_score(y_test_120, nn_scores)
svm_ap = average_precision_score(y_test_120, svm_scores)
knn_ap = average_precision_score(y_test_120, knn_scores)

# plotting combined precision-recall curve #
plt.figure()

plt.plot(
    nn_recall,
    nn_precision,
    label=f'Neural Network (AP = {nn_ap:.3f})'
)

plt.plot(
    svm_recall,
    svm_precision,
    label=f'SVM (AP = {svm_ap:.3f})'
)

plt.plot(
    knn_recall,
    knn_precision,
    label=f'k-NN (AP = {knn_ap:.3f})'
)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves for Final Models')
plt.legend()
plt.grid()
plt.show()

# printing average precision values #
print("\nAverage Precision Results")
print(f"Neural Network AP: {nn_ap:.4f}")
print(f"SVM AP: {svm_ap:.4f}")
print(f"k-NN AP: {knn_ap:.4f}")