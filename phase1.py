# 3.2.1: Loading & Merging the Data #
# importing libraries #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# creating function to read text files #
def parse_ph_file(filepath, label):
    records = []

# opening file to read each line #
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()

# continuing only if the 2 pieces (timestamp & pH) present
            if len(parts) == 2:
                try:
                    timestamp = int(parts[0])
                    ph = float(parts[1])

# storing each valid reading #
                    records.append({
                        'timestamp_ms': timestamp,
                        'ph_value': ph,
                        'label': label
                    })

                except ValueError:
                    pass

# after data is read, turn list of readings into a table #
    return pd.DataFrame(records)

# loading acid & non-acid datasets #
acid_df = parse_ph_file("acid_revised.txt", label=1)
nonacid_df = parse_ph_file("no_acid_revised.txt", label=0)

# displaying first 5 rows of both tables #
# print(acid_df.head())
# print(nonacid_df.head())
# remove hashtag if you want to display table later #

# combining both datasets into one dataframe #
df = pd.concat([acid_df, nonacid_df], ignore_index=True)

# checking number of records #
# print("Acid rows:", len(acid_df))
# print("Non-acid rows:", len(nonacid_df))
# print("Total rows:", len(df))

# displaying first 5 rows of combined table #
# print("\nCombined dataset:")
# print(df.head())


# 3.2.2: Exploratory Data Analysis #
# counting how many readings belong to each class #
class_counts = df['label'].value_counts().sort_index()

# print("\nClass distribution:")
# print(class_counts)

# ploting bar chart for class distribution #
# plt.figure(figsize=(6, 4))

# plt.bar(
    # ['Non-Acid Reflux (0)', 'Acid Reflux (1)'],
    # [class_counts[0], class_counts[1]]
# )

# plt.xlabel('Class')
# plt.ylabel('Number of Readings')
# plt.title('Class Distribution')

# plt.show()

# plotting histograms for pH distributions of each class #
# plt.figure(figsize=(8, 5))

# plt.hist(
    # nonacid_df['ph_value'],
    # bins=50,
    # alpha=0.6,
    # label='Non-Acid Reflux (0)'
# )

# plt.hist(
    # acid_df['ph_value'],
    # bins=50,
    # alpha=0.6,
    # label='Acid Reflux (1)'
# )

# plt.xlabel('pH Value')
# plt.ylabel('Frequency')
# plt.title('Distribution of pH Values by Reflux Class')
# plt.legend()

# plt.show()

# taking the first 500 readings from each class #
acid_sample = acid_df.head(500)
nonacid_sample = nonacid_df.head(500)

# plotting 500 acid reflux readings #
# plt.figure(figsize=(10, 5))

# plt.plot(
    # acid_sample['timestamp_ms'],
    # acid_sample['ph_value']
# )

# plt.xlabel('Timestamp (ms)')
# plt.ylabel('pH Value')
# plt.title('Acid Reflux: 500-Reading Time-Series Sample')

# plt.show()

# plotting 500 non-acid reflux readings #
# plt.figure(figsize=(10, 5))

# plt.plot(
    # nonacid_sample['timestamp_ms'],
    # nonacid_sample['ph_value']
# )

# plt.xlabel('Timestamp (ms)')
# plt.ylabel('pH Value')
# plt.title('Non-Acid Reflux: 500-Reading Time-Series Sample')

# plt.show()

# summary statistics for pH values in each class #

# separating combined dataset according to label & calculate useful statistics #
summary_stats = df.groupby('label')['ph_value'].describe()

# print("\nSummary statistics by class:")
# print(summary_stats)

# checking for missing values #
# print("\nMissing Values:")
# print(df.isnull().sum())

# checking minimum & maximum pH values #
# print("\npH Range:")
# print("Minimum pH:", df['ph_value'].min())
# print("Maximum pH:", df['ph_value'].max())

# counting number of unusually low pH readings #
low_ph_count = (df['ph_value'] < 2).sum()

# print("\nNumber of pH readings below 2:")
# print(low_ph_count)

# checking timestamp intervals for each dataset #
acid_intervals = acid_df['timestamp_ms'].diff()
nonacid_intervals = nonacid_df['timestamp_ms'].diff()

# print("\nAcid timestamp intervals:")
# print(acid_intervals.value_counts().head())

# print("\nNon-acid timestamp intervals:")
# print(nonacid_intervals.value_counts().head())


# 3.2.3: Feature Engineering #

# creating function to split pH data & calculate features for each window #
def extract_features(data, window_size):
    features = []

    # diving data into non-overlapping windows #
    for start in range(0, len(data) - window_size + 1, window_size):
        window = data.iloc[start:start + window_size]

        # getting only pH values from this window #
        ph = window['ph_value'].to_numpy()
        
        # mean pH #
        mean_ph = np.mean(ph)
        
        # standard deviation #
        std_ph = np.std(ph)
        
        # minimum pH #
        min_ph = np.min(ph)

        # maximum pH #
        max_ph = np.max(ph)

        # pH range #
        range_ph = max_ph - min_ph

        # median pH #
        median_ph = np.median(ph)

        # slope #
        time = np.arange(window_size)

        # drawing best fit line #
        slope = np.polyfit(time, ph, 1)[0]

        # above 7 ratio #
        above_7_ratio = np.mean(ph > 7.0)

        # below 6 ratio #
        below_6_ratio = np.mean(ph < 6.0)

        # majority class label in this window #
        label = window['label'].mode()[0]

        # storing features #
        features.append({
            'mean_ph': mean_ph,
            'std_ph': std_ph,
            'min_ph': min_ph,
            'max_ph': max_ph,
            'range_ph': range_ph,
            'median_ph': median_ph,
            'slope': slope,
            'above_7_ratio': above_7_ratio,
            'below_6_ratio': below_6_ratio,
            'label': label
        })

    # converting results into a dataframe #
    return pd.DataFrame(features)

# for window size 30 #
features_30 = extract_features(df, 30)

# print("\nwindow size = 30")
# print(features_30.head())

# print("\nShape:")
# print(features_30.shape)

# for window size 60 #
features_60 = extract_features(df, 60)

# print("\nwindow size = 60")
# print(features_60.head())

# print("\nShape:")
# print(features_60.shape)

# for window size 120 #
features_120 = extract_features(df, 120)

# print("\nwindow size = 120")
# print(features_120.head())

# print("\nShape:")
# print(features_120.shape)


# 3.2.4: Train/Test Split #

# for window size 30 #

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

# checking if stratified sampling worked #
# print("\nwindow size = 30")
# print("Training samples:", len(X_train_30))
# print("Test samples:", len(X_test_30))

# print("\nTraining class distribution:")
# print(y_train_30.value_counts())

# print("\nTest class distribution:")
# print(y_test_30.value_counts())

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
# checking if stratified sampling worked #
# print("\nwindow size = 60")
# print("Training samples:", len(X_train_60))
# print("Test samples:", len(X_test_60))

# print("\nTraining class distribution:")
# print(y_train_60.value_counts())

# print("\nTest class distribution:")
# print(y_test_60.value_counts())

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

# checking if stratified sampling worked #
# print("\nwindow size = 120")
# print("Training samples:", len(X_train_120))
# print("Test samples:", len(X_test_120))

# print("\nTraining class distribution:")
# print(y_train_120.value_counts())

# print("\nTest class distribution:")
# print(y_test_120.value_counts())

# Extra Preprocessing for Phase 2: Feature Scaling #

# for window size 30 #
# creating scaler #
scaler_30 = StandardScaler()

# learning scaling values from the training data & scale it #
X_train_30_scaled = scaler_30.fit_transform(X_train_30)

# scaling test data using the same scaling values so test data is not leaked #
X_test_30_scaled = scaler_30.transform(X_test_30)

# checking if scaling worked #
# print("\nFirst 5 rows BEFORE scaling:")
# print(X_train_30.head())

# print("\nFirst 5 rows AFTER scaling:")
# print(X_train_30_scaled[:5])

# for window size 60 #
scaler_60 = StandardScaler()

X_train_60_scaled = scaler_60.fit_transform(X_train_60)
X_test_60_scaled = scaler_60.transform(X_test_60)

# for window size 120 #
scaler_120 = StandardScaler()

X_train_120_scaled = scaler_120.fit_transform(X_train_120)
X_test_120_scaled = scaler_120.transform(X_test_120)

# saving processed datasets for phase 2 #
features_30.to_csv("features_30.csv", index=False)
features_60.to_csv("features_60.csv", index=False)
features_120.to_csv("features_120.csv", index=False)

print("\nProcessed window datasets saved successfully.")