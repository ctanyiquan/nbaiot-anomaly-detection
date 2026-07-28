"""Shared constants and helper functions for the N-BaIoT project."""

# Standard library
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Seed used everywhere so results are reproducible
RANDOM_SEED = 42

# Where each notebook stage reads and writes its files
RAW_DIRECTORY = "data/raw"
PROCESSED_DIRECTORY = "data/processed"
MODEL_DIRECTORY = "models"
FIGURE_DIRECTORY = "reports/figures"

CHOSEN_DEVICES = [
    # Chosen because all three have both Mirai and BASHLITE traffic,
    # are three different device types, and each has a healthy
    # benign baseline (smallest training split is about 34,700 rows).
    "Danmini_Doorbell",
    "Philips_B120N10_Baby_Monitor",
    "Provision_PT_838_Security_Camera",
    # Optional fourth: "Ecobee_Thermostat" adds a very different
    # device type but has only 13,113 benign rows (about 9,200 to
    # train on) - include it as an extra, not as a replacement, and
    # discuss the thin baseline in Findings.
]

# Per-device ceiling when balancing the pooled training and
# validation sets: every device contributes the same number of rows
# (the smallest contributing device's count), capped here only if
# that smallest count is unexpectedly large. Per-device models use
# all their own benign data.
POOLED_BENIGN_CAP = 50_000

# For the optional full-coverage summary run
ALL_DEVICES = [
    "Danmini_Doorbell",
    "Ecobee_Thermostat",
    "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor",
    "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera",
    "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
]

# These two devices only carry BASHLITE traffic, no Mirai
DEVICES_WITHOUT_MIRAI = ["Ennio_Doorbell", "Samsung_SNH_1011_N_Webcam"]

# Max rows kept per attack subtype when subsampling
ATTACK_CAP = 4000

# Drop one of any feature pair more correlated than this
CORRELATION_THRESHOLD = 0.95

# Keep enough PCA components to cover this much variance
PCA_VARIANCE = 0.95

# Alarm threshold = mean + Z * std of benign validation error
AUTOENCODER_THRESHOLD_Z = 3

# Alarm threshold = mean + Z * std of benign validation distance
KMEANS_THRESHOLD_Z = 3

# Meta columns describe a row but are never fed to a model
META_COLUMNS = ["device", "label", "attack_subtype"]


def label_from_path(csv_path):
    """Read the label and attack subtype from the file name."""
    stem = csv_path.stem.lower()
    if stem == "benign":
        return "benign", "none"
    return stem.split(".")[0], stem    # ("mirai", "mirai.ack")


def load_device(device_directory, device_name):
    """Load and label every CSV file for one device.

    Reads every CSV file found under device_directory (searched
    recursively), attaches "device", "label", and "attack_subtype"
    columns derived from each file's path, and concatenates the
    result into one dataframe.
    """
    # Search the device's folder, and any sub-folders, for CSVs
    device_directory = Path(device_directory)
    frames = []

    # Read each file and attach its label as three new columns
    for csv_path in sorted(device_directory.rglob("*.csv")):
        label, attack_subtype = label_from_path(csv_path)
        frame = pd.read_csv(csv_path)
        meta_columns = pd.DataFrame(
            {
                "device": device_name,
                "label": label,
                "attack_subtype": attack_subtype,
            },
            index=frame.index,
        )
        frames.append(pd.concat([frame, meta_columns], axis=1))

    # Stack every labelled file into one table for this device
    return pd.concat(frames, ignore_index=True)


def feature_columns(dataframe):
    """List the model-input feature columns of a labelled dataframe."""
    return [
        column for column in dataframe.columns
        if column not in META_COLUMNS
    ]


def subsample_device(dataframe, attack_cap=ATTACK_CAP, seed=RANDOM_SEED):
    """Stratified subsample for one device's labelled dataframe.

    Keeps all benign rows. Caps each attack subtype at attack_cap
    rows, sampled without replacement, reproducibly via seed.
    """
    # Split the rows into benign traffic and attack traffic
    benign_rows = dataframe[dataframe["label"] == "benign"]
    attack_rows = dataframe[dataframe["label"] != "benign"]

    # Randomly cap each attack subtype at attack_cap rows
    capped_groups = [
        group.sample(n=min(len(group), attack_cap), random_state=seed)
        for _, group in attack_rows.groupby("attack_subtype")
    ]

    # Combine the untouched benign rows with the capped attack rows
    return pd.concat([benign_rows] + capped_groups, ignore_index=True)


def split_benign_only(device_dataframe, seed=RANDOM_SEED):
    """Benign-only train/validation/test split for one device.

    Splits the benign rows 70/15/15 into train, validation, and
    test. Every attack row is appended to the test split only, so
    train and validation never contain an attack row. Returns
    (train, val, test) dataframes.
    """
    benign_rows = device_dataframe[device_dataframe["label"] == "benign"]
    attack_rows = device_dataframe[device_dataframe["label"] != "benign"]

    # Split off train first, then split the remainder into val and test
    train, remainder = train_test_split(
        benign_rows, train_size=0.7, random_state=seed
    )
    val, benign_test = train_test_split(
        remainder, train_size=0.5, random_state=seed
    )

    # Attack rows only ever appear in the test split
    test = pd.concat([benign_test, attack_rows], ignore_index=True)
    return train.reset_index(drop=True), val.reset_index(drop=True), test


def balance_rows_per_device(
    dataframe, ceiling=POOLED_BENIGN_CAP, seed=RANDOM_SEED
):
    """Sample an equal number of rows from every device.

    Each device contributes the same number of rows to a pooled
    split: the smallest device's group size, capped at ceiling.
    Sampling is without replacement and reproducible via seed, so
    no single device dominates the pooled set regardless of how
    much benign data it happens to have.
    """
    grouped = dataframe.groupby("device")
    per_device = min(grouped.size().min(), ceiling)

    # Take the same number of rows from each device
    balanced_groups = [
        group.sample(n=per_device, random_state=seed)
        for _, group in grouped
    ]
    return pd.concat(balanced_groups, ignore_index=True)


def prune_correlated_columns(
    dataframe, columns, threshold=CORRELATION_THRESHOLD
):
    """List columns to keep after dropping correlated duplicates.

    Computes the Pearson correlation matrix of the given columns.
    Of each pair correlated above threshold, the lower-variance
    column is dropped, so the more spread-out feature survives.
    Returns the surviving column names, in their original order.
    Callers should pass unscaled data, so variance is meaningful,
    and should exclude constant columns beforehand.
    """
    correlation_matrix = dataframe[columns].corr().abs()
    variance = dataframe[columns].var()
    ordered_by_variance = variance.sort_values(ascending=False).index

    # Go through list of columns from highest to lowest variance and
    # keep a column unless it is too correlated with a higher variance
    # column that is already kept
    kept_columns = set()
    for column in ordered_by_variance:
        too_correlated = any(
            correlation_matrix.loc[column, kept] > threshold
            for kept in kept_columns
        )
        if not too_correlated:
            kept_columns.add(column)

    return [column for column in columns if column in kept_columns]


def save_arrays(device, **arrays):
    """Save one or more named arrays to a device's processed folder.

    Writes each keyword array to
    data/processed/<device>/<name>.npy, creating the folder first
    if it does not already exist.
    """
    device_directory = Path(PROCESSED_DIRECTORY) / device
    device_directory.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(device_directory / f"{name}.npy", array)
