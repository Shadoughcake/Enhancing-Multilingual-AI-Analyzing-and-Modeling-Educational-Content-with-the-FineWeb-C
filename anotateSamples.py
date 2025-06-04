import numpy as np
import pandas as pd
import tqdm as tqdm
import ast

# Load dataset
DATASET = pd.read_parquet("hf://datasets/data-is-better-together/fineweb-c/dan_Latn/train-00000-of-00001.parquet")
pd.set_option("display.max_colwidth", None)

# Prepare DataFrame
df = pd.DataFrame()
df["text"] = DATASET["text"]
df["id"] = DATASET.index

# Define labels
labels = ["None", "Minimal", "Basic", "Good", "Excellent"]

def save_annotation(sample, path):
    sample_id = sample["id"].values[0]
    sample_text = sample["text"].values[0]
    new_label = sample["label"].values[0]

    # Try to load existing annotations
    try:
        existing = pd.read_csv(path)
        existing["label"] = existing["label"].apply(ast.literal_eval)  # Convert label strings to lists
    except FileNotFoundError:
        # No file yet, create new DataFrame with label as list
        new_df = pd.DataFrame([{
            "id": sample_id,
            "text": sample_text,
            "label": [new_label]
        }])
        new_df.to_csv(path, index=False)
        return

    # Check if this sample already exists
    match = existing[existing["id"] == sample_id]

    if not match.empty:
        idx = match.index[0]
        current_labels = existing.at[idx, "label"]
        current_labels.append(new_label)
        existing.at[idx, "label"] = current_labels
    else:
        new_row = pd.DataFrame([{
            "id": sample_id,
            "text": sample_text,
            "label": [new_label]
        }])
        existing = pd.concat([existing, new_row], ignore_index=True)

    existing.to_csv(path, index=False)


current_sample = None  # Holds the last sample if input was invalid

while True:
    if current_sample is None:
        sample = df.sample(n=1)
    else:
        sample = current_sample  # Reuse previous sample

    text = sample["text"].values[0]
    print("\n" + "="*80)
    print("TEXT SAMPLE:\n")
    print(text)
    print("="*80)

    user_input = input("\nEnter the label [0: None, 1: Minimal, 2: Basic, 3: Good, 4: Excellent], 's' to skip, or 'q' to quit: ").strip().lower()

    if user_input == 'q':
        break

    elif user_input == 's':
        print("Skipping this sample.")
        current_sample = None  # move on

    elif user_input in ['0', '1', '2', '3', '4']:
        label_index = int(user_input)
        label = labels[label_index]
        sample["label"] = label
        print(f"Label assigned: {label}")
        save_annotation(sample, "annotations.csv")
        current_sample = None  # move on

    else:
        print("Invalid input. Please try again — same sample will be shown.")
        current_sample = sample  # keep the same one next loop