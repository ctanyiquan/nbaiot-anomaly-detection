"""Shared constants and helper functions for the N-BaIoT project."""

# Standard library
from pathlib import Path

# Third-party
import pandas

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

# Cap benign rows per device when building the pooled training set
# only, so no single device dominates the shared baseline. Per-device
# models use all their own benign data.
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
        frame = pandas.read_csv(csv_path)
        meta_columns = pandas.DataFrame(
            {
                "device": device_name,
                "label": label,
                "attack_subtype": attack_subtype,
            },
            index=frame.index,
        )
        frames.append(pandas.concat([frame, meta_columns], axis=1))

    # Stack every labelled file into one table for this device
    return pandas.concat(frames, ignore_index=True)


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
    return pandas.concat([benign_rows] + capped_groups, ignore_index=True)
