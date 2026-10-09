# Gastroesophageal Reflux (GERD) Classification from Oesophageal pH Recordings

Supervised learning assignment: binary classification of **acid reflux (1)** vs. **non-acid reflux (0)**
from oesophageal pH sensor time-series, comparing **Neural Networks (Keras/TensorFlow)**,
**Support Vector Machines** and **k-Nearest Neighbours** (scikit-learn).

All experiments are reproducible on a standard Linux machine (Ubuntu 20.04+).

## Repository Layout

| File | Description |
| --- | --- |
| `acid_revised.txt`, `no_acid_revised.txt` | Raw pH sensor recordings — one `TimeStamp(ms)  Ch 1 (pH)` pair per line, one file per class |
| `phase1.py` | Data loading & merging, EDA, windowed feature engineering, stratified train/test split; writes the `features_*.csv` files |
| `phase2_nn.py` | Neural network experiments NN-1 … NN-4 (depth, activation, dropout/L2), Adam vs. SGD, window-size comparison |
| `phase2_svm.py` | SVM experiments: linear + polynomial kernels, RBF `C`/`gamma` grid, window-size comparison |
| `phase2_knn.py` | k-NN experiments: `k` grid × distance metric × weighting, window-size comparison |
| `phase3.py` | Final evaluation of the best model per algorithm on the held-out test set: confusion matrices, ROC curves, precision-recall curves |
| `features_30.csv`, `features_60.csv`, `features_120.csv` | Windowed feature datasets produced by `phase1.py` (pre-generated copies included) |
| `requirements.txt` | Pinned Python dependencies |

## Requirements

- **OS**: standard Linux (tested against Ubuntu 20.04+); no Windows-specific paths are used.
- **Python**: 3.12 recommended (TensorFlow 2.19 supports Python 3.9–3.12; there are no 3.13+ wheels).
- **Packages**: pinned versions in `requirements.txt` — numpy, pandas, matplotlib, seaborn, scikit-learn, scipy, tensorflow, setuptools.

## Setup

From the project root:

```bash
# create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# install the pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Ubuntu 24.04 ships Python 3.12 by default (`python3 -m venv .venv` works).
On Ubuntu 20.04/22.04, install Python 3.12 first, e.g.:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv
```

## Running the Experiments

All scripts use relative paths — **run them from the project root**:

```bash
# 1. Data preparation (fast, < 1 min)
#    Parses the raw recordings, engineers windowed features for window sizes
#    30/60/120, and writes features_30.csv, features_60.csv, features_120.csv.
python phase1.py

# 2. Algorithm experimentation — independent of each other, but they require the
#    features_*.csv files (pre-generated copies are included in this repository).
python phase2_nn.py     # 5-fold CV: NN-1..NN-4, Adam vs. SGD, window sizes — slowest (~30–60 min on CPU)
python phase2_svm.py    # linear/polynomial kernels + RBF C/gamma grid + window sizes (~2–5 min)
python phase2_knn.py    # k/metric/weighting grid + window-size comparison (< 1 min)

# 3. Final evaluation (a few minutes)
#    Trains the final NN (NN-4, Adam), SVM (RBF, C=100, gamma=0.01) and k-NN
#    (k=25, Manhattan, distance weighting) on the full training set and evaluates
#    them once on the held-out test set.
python phase3.py
```

### Notes

- Re-running `phase1.py` regenerates the `features_*.csv` files from the raw data
  (the committed copies were produced by exactly this script).
- Figures are displayed with `plt.show()`. On a headless server, either run from a
  desktop session / with X11 forwarding, or set `MPLBACKEND=Agg` — figures are then
  not displayed and `phase3.py` prints `FigureCanvasAgg is non-interactive` warnings
  (expected, harmless):

  ```bash
  MPLBACKEND=Agg python phase3.py
  ```

- `phase3.py` opens several figure windows (3 confusion matrices, 1 ROC plot,
  1 precision-recall plot); close each window to continue when using an interactive backend.

## Reproducibility

- All random seeds are fixed to `42` (Python `random`, NumPy, TensorFlow), and the
  train/test split is stratified with `random_state=42` in every script.
- `phase1.py` is deterministic; the committed CSVs match regenerated output up to
  last-bit floating-point rounding in the `slope` feature (differences between NumPy builds).
- Cross-validation is 5-fold `StratifiedKFold(shuffle=True, random_state=42)` on the
  training set only; the test set is used once in `phase3.py`.

## Expected Results (reference run, CPU, pinned versions)

`phase1.py` prints one line on success (all EDA plots are kept commented out in the script):

```
Processed window datasets saved successfully.
```

Final test-set results printed by `phase3.py`:

```
Final Neural Network Test Results
Test Accuracy: 0.8368
Test Loss:     0.3992

Final SVM Test Results
Test Accuracy: 0.8333

Final k-NN Test Results
Test Accuracy: 0.8194

AUC Results
Neural Network AUC: 0.9067
SVM AUC:            0.9079
k-NN AUC:           0.9093
```

Window-size comparison printed by `phase2_knn.py` (k = 25, Manhattan, distance weighting):

```
Window size = 30   Mean CV accuracy: 0.8075
Window size = 60   Mean CV accuracy: 0.8213
Window size = 120  Mean CV accuracy: 0.8391
```

RBF SVM grid printed by `phase2_svm.py` — best cell: `C=100, gamma=0.01, Mean CV Accuracy=0.8012`.

Neural-network results may vary in the last decimal place between platforms/runs due to
non-deterministic floating-point reductions; accuracies are stable to roughly ±0.01.

## Troubleshooting

- `ModuleNotFoundError: No module named 'setuptools.dist'` — the virtual environment is
  not active, or `pip install -r requirements.txt` was not run (`phase2_nn.py` and
  `phase3.py` import `setuptools.dist`).
- `pip` cannot find a matching TensorFlow wheel — your Python is newer than 3.12;
  use Python 3.9–3.12.
- Tkinter/display errors when plotting — set `MPLBACKEND=Agg` (see Notes above).
