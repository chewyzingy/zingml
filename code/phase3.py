# 3.4 Final Model Evaluation #

# importing libraries #
import os as _os

# used to time how long each final model takes to train #
import time

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

# helper: turning a model name into a filename friendly token #
def slug(text):
    # 'Neural Network' -> 'neural_network', 'k-NN' -> 'k_nn' #
    cleaned = text.lower().replace('-', ' ').replace('_', ' ')

    return '_'.join(cleaned.split())

# setting random seed for reproducibility #
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# loading selected window size dataset #
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

features_120 = pd.read_csv(data_path("features_120.csv"))

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

# adding early stopping #
# a tenth of the training set is held back as the validation set & training stops
# once the validation loss has not improved for 10 consecutive epochs #
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# training final neural network model on the training set #
# (one validation split is carved out of the training set by Keras, so the
# held-out test set is still only touched by the evaluate call below) #
start_time = time.perf_counter()

nn_history = final_nn.fit(
    X_train_120_scaled,
    y_train_120,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=0
)

nn_training_time = time.perf_counter() - start_time

# reporting how long the neural network took to train #
print("\nFinal Neural Network Training")
print(f"Epochs run: {len(nn_history.history['loss'])} (out of 100, early stopping)")
print(f"Training time: {nn_training_time:.2f} seconds")

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
start_time = time.perf_counter()

final_svm.fit(
    X_train_120_scaled,
    y_train_120
)

svm_training_time = time.perf_counter() - start_time

# evaluating final SVM on held-out test set #
svm_test_accuracy = final_svm.score(
    X_test_120_scaled,
    y_test_120
)

print("\nFinal SVM Test Results")
print(f"Test Accuracy: {svm_test_accuracy:.4f}")
print(f"Training time: {svm_training_time:.2f} seconds")

# final kNN model #
# using k = 25, Manhattan distance, & distance weighting #
# using window size 120 as it achieved the highest CV accuracy #

final_knn = KNeighborsClassifier(
    n_neighbors=25,
    metric='manhattan',
    weights='distance'
)

# training final k-NN on full training set #
start_time = time.perf_counter()

final_knn.fit(
    X_train_120_scaled,
    y_train_120
)

knn_training_time = time.perf_counter() - start_time

# evaluating final k-NN on held-out test set #
knn_test_accuracy = final_knn.score(
    X_test_120_scaled,
    y_test_120
)

print("\nFinal k-NN Test Results")
print(f"Test Accuracy: {knn_test_accuracy:.4f}")
print(f"Training time: {knn_training_time:.2f} seconds")

# comparing training duration of the three final models #
print("\nTraining Time Comparison")
print(f"SVM training time: {svm_training_time:.2f} seconds")
print(f"k-NN training time: {knn_training_time:.2f} seconds")
print(f"Neural Network training time: {nn_training_time:.2f} seconds")

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
    save_fig(f"phase3_confusion_matrix_{slug(model_name)}.png")
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
save_fig("phase3_roc_curves.png")
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
save_fig("phase3_precision_recall_curves.png")
plt.show()

# printing average precision values #
print("\nAverage Precision Results")
print(f"Neural Network AP: {nn_ap:.4f}")
print(f"SVM AP: {svm_ap:.4f}")
print(f"k-NN AP: {knn_ap:.4f}")
