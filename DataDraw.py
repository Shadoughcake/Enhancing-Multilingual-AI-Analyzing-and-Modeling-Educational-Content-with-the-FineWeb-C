import numpy as np
import pandas as pd
import tqdm as tqdm
import ast

# Load dataset
DATASET = pd.read_parquet("hf://datasets/data-is-better-together/fineweb-c/dan_Latn/train-00000-of-00001.parquet")

pd.set_option("display.max_colwidth", None)

# Prepare DataFrame
# ------------------------------------------------------------
# We take a *deterministic* subset of 50 rows so that every
# annotator sees the exact same pieces of text in the same order.
# ------------------------------------------------------------

df = pd.DataFrame()
df["text"] = DATASET["text"]
df["id"] = DATASET.index

# Fixed subset & ordering
SUBSET_SIZE = 50
RANDOM_SEED = 43  # any constant value works – keep it unchanged!

df = df.sample(n=SUBSET_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)

# ----------------------------------------------------------------------------
# Annotation settings
# ----------------------------------------------------------------------------
labels = [
    "None",       # 0
    "Minimal",    # 1
    "Basic",      # 2
    "Good",       # 3
    "Excellent",  # 4
    "Problematic" # 's' – special case
]


def save_annotation(sample: pd.DataFrame, path: str) -> None:
    """Append or update a sample's annotation in *path*.

    If the sample already exists we append the new label to its list so that we
    can later compute inter‑annotator agreement.
    """

    sample_id = sample["id"].values[0]
    sample_text = sample["text"].values[0]
    new_label = sample["label"].values[0]

    # Try to load existing annotations
    try:
        existing = pd.read_csv(path)
        existing["label"] = existing["label"].apply(ast.literal_eval)  # Convert label strings → lists
    except FileNotFoundError:
        # No annotation file yet – create it
        pd.DataFrame([
            {"id": sample_id, "text": sample_text, "label": [new_label]}
        ]).to_csv(path, index=False)
        return

    # Check if this sample already exists
    match = existing[existing["id"] == sample_id]

    if not match.empty:
        idx = match.index[0]
        current_labels: list = existing.at[idx, "label"]
        current_labels.append(new_label)
        existing.at[idx, "label"] = current_labels
    else:
        # New sample – append to CSV
        new_row = pd.DataFrame([
            {"id": sample_id, "text": sample_text, "label": [new_label]}
        ])
        existing = pd.concat([existing, new_row], ignore_index=True)

    existing.to_csv(path, index=False)


# ----------------------------------------------------------------------------
# Interactive annotation loop
# ----------------------------------------------------------------------------
current_sample = None      # Holds the last sample if input was invalid
sample_index = 0           # Where we are in the deterministic subset
TOTAL_SAMPLES = len(df)    # 50

print("\nLoaded", TOTAL_SAMPLES, "deterministic samples for annotation.")
print("Press 'q' at any time to quit.\n")

while True:
    # --------------------------------------------------------
    # Select next sample (or reuse the previous one if needed)
    # --------------------------------------------------------
    if current_sample is None:
        sample = df.iloc[[sample_index]]
        sample_index = (sample_index + 1) % TOTAL_SAMPLES  # cycles when we wrap around
    else:
        sample = current_sample  # Re‑show same sample after invalid input

    # --------------------------------------------------------
    # Display sample to user
    # --------------------------------------------------------
    text = sample["text"].values[0]
    print("\n" + "="*80)
    print("TEXT SAMPLE (ID:", sample["id"].values[0], ")\n")
    print(text)
    print("="*80)

    # --------------------------------------------------------
    # Collect user input
    # --------------------------------------------------------
    user_input = input(
        "\nEnter the label [0: None, 1: Minimal, 2: Basic, 3: Good, 4: Excellent], 's' for Problematic, or 'q' to quit: "
    ).strip().lower()

    # --------------------------------------------------------
    # Act on user input
    # --------------------------------------------------------
    if user_input == 'q':
        print("Exiting. Goodbye!")
        break

    elif user_input == 's':
        # Mark sample as problematic (special category)
        label = "Problematic"
        sample["label"] = label
        print(f"Label assigned: {label}")
        save_annotation(sample, "annotations.csv")
        current_sample = None  # move on to next text

    elif user_input in ['0', '1', '2', '3', '4']:
        label_index = int(user_input)
        label = labels[label_index]
        sample["label"] = label
        print(f"Label assigned: {label}")
        save_annotation(sample, "annotations.csv")
        current_sample = None  # move on to next text

    else:
        # Any other key – input was invalid, re‑show same text
        print("Invalid input. Please try again — the same sample will be shown.")
        current_sample = sample

