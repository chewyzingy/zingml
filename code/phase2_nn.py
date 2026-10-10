# importing libraries #
import os as _os

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split, StratifiedKFold
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
tf.random.set_seed(SEED)

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

print("30:", features_30.shape)
print("60:", features_60.shape)
print("120:", features_120.shape)

# Data Preparation Before Start of Phase 2 NN #
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
print("30-reading data prepared successfully")

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
print("60-reading data prepared successfully")

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
print("120-reading data prepared successfully")


# 3.3.1: Neural Networks #

# NN-1: Single Hidden Layer, ReLU, No Regularisation #
model_nn1 = keras.Sequential([
    layers.Input(shape=(9,)),
    layers.Dense(7, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_nn1.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_nn1.summary()

# 5 fold cross validation for NN-1 #
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_accuracies_nn1 = []
cv_histories_nn1 = []

# separating each fold into fold training data & fold validation data #
for fold, (train_index, val_index) in enumerate(
    skf.split(X_train_30_scaled, y_train_30), start=1
):
    print(f"\nTraining Fold {fold}...")

    X_fold_train = X_train_30_scaled[train_index]
    X_fold_val = X_train_30_scaled[val_index]

    y_train_array = y_train_30.to_numpy()
    y_fold_train = y_train_array[train_index]
    y_fold_val = y_train_array[val_index]

    # creating a fresh NN-1 for this fold #
    # to ensure independent training across all folds #
    fold_model = keras.Sequential([
        layers.Input(shape=(9,)),
        layers.Dense(7, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    fold_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # adding early stopping #
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    history = fold_model.fit(
        X_fold_train,
        y_fold_train,
        validation_data=(X_fold_val, y_fold_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    cv_histories_nn1.append(history.history)

    # reporting training validation loss & accuracy for each fold #
    val_loss, val_accuracy = fold_model.evaluate(
        X_fold_val,
        y_fold_val,
        verbose=0
   )

    cv_accuracies_nn1.append(val_accuracy)

    print(f"Fold {fold} Validation Accuracy: {val_accuracy:.4f}")

print("\nNN-1 5-Fold Cross-Validation Results")
print(f"Mean Validation Accuracy: {np.mean(cv_accuracies_nn1):.4f}")
print(f"Standard Deviation: {np.std(cv_accuracies_nn1):.4f}")

# creating training validation loss & accuracy curves for NN-1 #
for i, fold_history in enumerate(cv_histories_nn1, start=1):

    # accuracy curves #
    plt.figure()
    plt.plot(fold_history['accuracy'], label='Training Accuracy')
    plt.plot(fold_history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'NN-1 Fold {i} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    save_fig(f"phase2_nn1_fold_{i}_accuracy.png")
    plt.show()

    # loss curves #
    plt.figure()
    plt.plot(fold_history['loss'], label='Training Loss')
    plt.plot(fold_history['val_loss'], label='Validation Loss')
    plt.title(f'NN-1 Fold {i} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    save_fig(f"phase2_nn1_fold_{i}_loss.png")
    plt.show()

# NN-2: Two Hidden Layers, ReLU, Dropout 0.3 #
model_nn2 = keras.Sequential([
    layers.Input(shape=(9,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model_nn2.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_nn2.summary()

# 5 fold cross validation for NN-2 #
cv_accuracies_nn2 = []
cv_histories_nn2 = []

# separating each fold into fold training data & fold validation data #
for fold, (train_index, val_index) in enumerate(
    skf.split(X_train_30_scaled, y_train_30), start=1
):
    print(f"\nTraining NN-2 Fold {fold}...")

    X_fold_train = X_train_30_scaled[train_index]
    X_fold_val = X_train_30_scaled[val_index]

    y_train_array = y_train_30.to_numpy()
    y_fold_train = y_train_array[train_index]
    y_fold_val = y_train_array[val_index]

    # creating a fresh NN-2 for this fold
    # to ensure independent training across all folds
    fold_model_nn2 = keras.Sequential([
        layers.Input(shape=(9,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    fold_model_nn2.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # adding early stopping #
    early_stopping_nn2 = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history_nn2 = fold_model_nn2.fit(
        X_fold_train,
        y_fold_train,
        validation_data=(X_fold_val, y_fold_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping_nn2],
        verbose=0
    )

    cv_histories_nn2.append(history_nn2.history)
    
    # reporting training validation loss & accuracy for each fold #
    val_loss, val_accuracy = fold_model_nn2.evaluate(
        X_fold_val,
        y_fold_val,
        verbose=0
    )

    cv_accuracies_nn2.append(val_accuracy)

    print(f"Fold {fold} Validation Accuracy: {val_accuracy:.4f}")
    
print("\nNN-2 5-Fold Cross-Validation Results")
print(f"Mean Validation Accuracy: {np.mean(cv_accuracies_nn2):.4f}")
print(f"Standard Deviation: {np.std(cv_accuracies_nn2):.4f}")

# creating training validation loss & accuracy curves for NN-2 #
for i, fold_history in enumerate(cv_histories_nn2, start=1):

    # accuracy curves #
    plt.figure()
    plt.plot(fold_history['accuracy'], label='Training Accuracy')
    plt.plot(fold_history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'NN-2 Fold {i} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    save_fig(f"phase2_nn2_fold_{i}_accuracy.png")
    plt.show()

    # loss curves #
    plt.figure()
    plt.plot(fold_history['loss'], label='Training Loss')
    plt.plot(fold_history['val_loss'], label='Validation Loss')
    plt.title(f'NN-2 Fold {i} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    save_fig(f"phase2_nn2_fold_{i}_loss.png")
    plt.show()

# NN-3: Three Hidden Layers, Tanh, L2 Regularisation #
model_nn3 = keras.Sequential([
    layers.Input(shape=(9,)),
    layers.Dense(16, activation='tanh',
                 kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dense(8, activation='tanh',
                 kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dense(4, activation='tanh',
                 kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dense(1, activation='sigmoid')
])

model_nn3.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_nn3.summary()

# 5 fold cross validation for NN-3 #
cv_accuracies_nn3 = []
cv_histories_nn3 = []

# separating each fold into fold training data & fold validation data #
for fold, (train_index, val_index) in enumerate(
    skf.split(X_train_30_scaled, y_train_30), start=1
):
    print(f"\nTraining NN-3 Fold {fold}...")

    X_fold_train = X_train_30_scaled[train_index]
    X_fold_val = X_train_30_scaled[val_index]

    y_train_array = y_train_30.to_numpy()
    y_fold_train = y_train_array[train_index]
    y_fold_val = y_train_array[val_index]

    # creating a fresh NN-3 for this fold #
    # to ensure independent training across all folds #
    fold_model_nn3 = keras.Sequential([
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

    fold_model_nn3.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # adding early stopping #
    early_stopping_nn3 = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history_nn3 = fold_model_nn3.fit(
        X_fold_train,
        y_fold_train,
        validation_data=(X_fold_val, y_fold_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping_nn3],
        verbose=0
    )

    cv_histories_nn3.append(history_nn3.history)
    
    # reporting training validation loss & accuracy for each fold #
    val_loss, val_accuracy = fold_model_nn3.evaluate(
        X_fold_val,
        y_fold_val,
        verbose=0
    )

    cv_accuracies_nn3.append(val_accuracy)

    print(f"Fold {fold} Validation Accuracy: {val_accuracy:.4f}")
    
print("\nNN-3 5-Fold Cross-Validation Results")
print(f"Mean Validation Accuracy: {np.mean(cv_accuracies_nn3):.4f}")
print(f"Standard Deviation: {np.std(cv_accuracies_nn3):.4f}")

# creating training validation loss & accuracy curves for NN-3 #
for i, fold_history in enumerate(cv_histories_nn3, start=1):

    # accuracy curves #
    plt.figure()
    plt.plot(fold_history['accuracy'], label='Training Accuracy')
    plt.plot(fold_history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'NN-3 Fold {i} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    save_fig(f"phase2_nn3_fold_{i}_accuracy.png")
    plt.show()

    # loss curves #
    plt.figure()
    plt.plot(fold_history['loss'], label='Training Loss')
    plt.plot(fold_history['val_loss'], label='Validation Loss')
    plt.title(f'NN-3 Fold {i} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    save_fig(f"phase2_nn3_fold_{i}_loss.png")
    plt.show()

# NN-4: Best Architecture from NN-1 to NN-3, Adam vs SGD #
# (the comparison itself builds a fresh model per fold below, so no top level
# NN-4 model object is needed here) #

# 5 fold cross validation for NN-4 Adam vs SGD #
cv_accuracies_nn4_adam = []
cv_accuracies_nn4_sgd = []

cv_histories_nn4_adam = []
cv_histories_nn4_sgd = []

# separating each fold into fold training data & fold validation data #
for fold, (train_index, val_index) in enumerate(
    skf.split(X_train_30_scaled, y_train_30), start=1
):
    print(f"\nTraining NN-4 Fold {fold}...")

    X_fold_train = X_train_30_scaled[train_index]
    X_fold_val = X_train_30_scaled[val_index]

    y_train_array = y_train_30.to_numpy()
    y_fold_train = y_train_array[train_index]
    y_fold_val = y_train_array[val_index]

    # NN-4 with Adam #
    fold_model_adam = keras.Sequential([
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

    fold_model_adam.compile(
            optimizer=keras.optimizers.Adam(),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    # adding early stopping for Adam #
    early_stopping_adam = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # training Adam model #
    history_adam = fold_model_adam.fit(
        X_fold_train,
        y_fold_train,
        validation_data=(X_fold_val, y_fold_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping_adam],
        verbose=0
    )

    # saving Adam training history #
    cv_histories_nn4_adam.append(history_adam.history)

    # reporting Adam training validation loss & accuracy for each fold #
    val_loss_adam, val_accuracy_adam = fold_model_adam.evaluate(
            X_fold_val,
            y_fold_val,
            verbose=0
        )

    # saving validation accuracy #
    cv_accuracies_nn4_adam.append(val_accuracy_adam)
    
    print(f"Adam Validation Accuracy: {val_accuracy_adam:.4f}")

    # NN-4 with SGD #
    fold_model_sgd = keras.Sequential([
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

    fold_model_sgd.compile(
        optimizer=keras.optimizers.SGD(
            learning_rate=0.01,
            momentum=0.9
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # adding early stopping for SGD #
    early_stopping_sgd = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
    )

    # training SGD model #
    history_sgd = fold_model_sgd.fit(
            X_fold_train,
            y_fold_train,
            validation_data=(X_fold_val, y_fold_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping_sgd],
            verbose=0
    )

    # saving SGD training history #
    cv_histories_nn4_sgd.append(history_sgd.history)

    # reporting SGD training validation loss & accuracy for each fold #
    val_loss_sgd, val_accuracy_sgd = fold_model_sgd.evaluate(
            X_fold_val,
            y_fold_val,
            verbose=0
    )

    # saving validation accuracy #
    cv_accuracies_nn4_sgd.append(val_accuracy_sgd)

    print(f"SGD Validation Accuracy: {val_accuracy_sgd:.4f}")

# NN-4 overall cross validation results #
print("\nNN-4 Adam vs SGD 5-Fold Cross-Validation Results")

print("\nAdam:")
print(f"Mean Validation Accuracy: {np.mean(cv_accuracies_nn4_adam):.4f}")
print(f"Standard Deviation: {np.std(cv_accuracies_nn4_adam):.4f}")

print("\nSGD + Momentum:")
print(f"Mean Validation Accuracy: {np.mean(cv_accuracies_nn4_sgd):.4f}")
print(f"Standard Deviation: {np.std(cv_accuracies_nn4_sgd):.4f}")

# creating training validation loss & accuracy curves for NN-4 Adam #
for i, fold_history in enumerate(cv_histories_nn4_adam, start=1):

    # accuracy curves #
    plt.figure()
    plt.plot(fold_history['accuracy'], label='Training Accuracy')
    plt.plot(fold_history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'NN-4 Adam Fold {i} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    save_fig(f"phase2_nn4_adam_fold_{i}_accuracy.png")
    plt.show()

    # loss curves #
    plt.figure()
    plt.plot(fold_history['loss'], label='Training Loss')
    plt.plot(fold_history['val_loss'], label='Validation Loss')
    plt.title(f'NN-4 Adam Fold {i} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    save_fig(f"phase2_nn4_adam_fold_{i}_loss.png")
    plt.show()

# creating training validation loss & accuracy curves for NN-4 SGD #
for i, fold_history in enumerate(cv_histories_nn4_sgd, start=1):

    # accuracy curves #
    plt.figure()
    plt.plot(fold_history['accuracy'], label='Training Accuracy')
    plt.plot(fold_history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'NN-4 SGD Fold {i} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    save_fig(f"phase2_nn4_sgd_fold_{i}_accuracy.png")
    plt.show()

    # loss curves #
    plt.figure()
    plt.plot(fold_history['loss'], label='Training Loss')
    plt.plot(fold_history['val_loss'], label='Validation Loss')
    plt.title(f'NN-4 SGD Fold {i} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    save_fig(f"phase2_nn4_sgd_fold_{i}_loss.png")
    plt.show()

# comparing effect of window size on NN performance #
# using NN-4 architecture with Adam as it achieved the highest accuracy #
# architecture & optimizer are constant, only window size changes #
nn_window_sizes = [60, 120]

X_train_nn_sets = [
    X_train_60_scaled,
    X_train_120_scaled
]

y_train_nn_sets = [
    y_train_60,
    y_train_120
]

for window_size, X_data, y_data in zip(
    nn_window_sizes,
    X_train_nn_sets,
    y_train_nn_sets
):

    fold_accuracies = []

    print(f"\nWindow size = {window_size}")

    # 5-fold cross validation #
    for fold, (train_index, val_index) in enumerate(
        skf.split(X_data, y_data), start=1
    ):

        X_fold_train = X_data[train_index]
        X_fold_val = X_data[val_index]

        y_fold_train = y_data.iloc[train_index]
        y_fold_val = y_data.iloc[val_index]

        # NN-4 with Adam #
        fold_model_adam = keras.Sequential([
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

        fold_model_adam.compile(
            optimizer=keras.optimizers.Adam(),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # adding early stopping for Adam #
        early_stopping_adam = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # training Adam model #
        fold_model_adam.fit(
            X_fold_train,
            y_fold_train,
            validation_data=(X_fold_val, y_fold_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping_adam],
            verbose=0
        )

        # reporting validation accuracy for each fold #
        val_loss_adam, val_accuracy_adam = fold_model_adam.evaluate(
            X_fold_val,
            y_fold_val,
            verbose=0
        )

        fold_accuracies.append(val_accuracy_adam)

        print(f"Fold {fold}: {val_accuracy_adam:.4f}")

    print(f"Mean CV accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Standard deviation: {np.std(fold_accuracies):.4f}")
