# Importing stock ML Libraries
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig, set_seed
from sklearn.model_selection import train_test_split
from collections import Counter
import ast
import matplotlib.pyplot as plt
import json

# Sections of config

# Defining some key variables that will be used later on in the training
############

Binary_Classification = False  # Set to True for binary classification, False for multiclass
MAX_LEN = 128*4
TRAIN_SIZE = 0.8
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 1e-06
DROPOUT = 0.5
LOSS_FUNCTION = "l1"  # Options: "weighted", "l1", "l1+weighted", "None"

SEED = 200

np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED) 

### Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained("Maltehb/danish-bert-botxo")
BERTmodel = BertModel.from_pretrained("Maltehb/danish-bert-botxo")

### Output file names
param_name = "L1" # Options: L1, L1Weighted, Weighted

graphname = "DK_final"+param_name+".png"
jsonName = "CMS_final"+param_name+".json"
misclassifiedName = "MC_final"+param_name+".csv"
############

print("Running:", graphname)

# # Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)




def Most_common_label(label_list):
    label = Counter(label_list).most_common(1)[0][0]
    inx = sort_order[label]  # Most frequent label
    L = [0 for i in range(len(unique_labels))]
    L[inx] = 1
    return L


def Soft_label(label_list):
    numeric_annotations = [sort_order[label] for label in label_list]
    return np.bincount(numeric_annotations, minlength=len(unique_labels)) / len(label_list)
    

#########
# DATASET = pd.read_parquet("hf://datasets/data-is-better-together/fineweb-c/dan_Latn/train-00000-of-00001.parquet")
DATASET = pd.read_csv("fineweb-c_relabled.csv")
DATASET["educational_value_labels"] = DATASET["educational_value_labels"].apply(ast.literal_eval)

PROBLEMATIC_CONTENT = False
LABEL_FUNCTION = Most_common_label
ex_Data_path = "fineweb2_data.csv"
#########


df = pd.DataFrame()
df["text"] = DATASET["text"]
df["educational_value_labels"] = DATASET["educational_value_labels"]
df["problematic_content_label_present"] = DATASET["problematic_content_label_present"]

extra_df = pd.read_csv(ex_Data_path)
extra_df = extra_df.rename(columns={"label": "educational_value_labels"})
extra_df['educational_value_labels'] = extra_df['educational_value_labels'].apply(ast.literal_eval)
extra_df['problematic_content_label_present'] = False

df = pd.concat([df, extra_df], ignore_index=True)

# REMOVE PROBLEMATIC LABELS FROM DATASET
df = df[df['problematic_content_label_present'] == PROBLEMATIC_CONTENT]

unique_labels = df["educational_value_labels"].explode().unique().tolist()
sort_order = {
    "None": 0,
    "Minimal": 1,
    "Basic": 2,
    "Good": 3,
    "Excellent": 4
}

# Process Data labels
df["Final_label"] = df["educational_value_labels"].apply(LABEL_FUNCTION)

# Convert Multi-Labels to Binary Labels
if Binary_Classification:
    unique_labels = ["None", "Educational"]
    sort_order = {
        "None": 0,
        "Educational": 1
    }
    df["educational_value_labels"] = df["educational_value_labels"].apply(
        lambda x: ["Educational"] if "None" not in x else ["None"])
    
    df["Final_label"] = df["Final_label"].apply(
        lambda x: [0,1] if "Educational" in x else [1,0]
    )

# Process Data labels
df["Final_label"] = df["educational_value_labels"].apply(LABEL_FUNCTION)

# Display sample rows
#df.sample(5)


new_df = pd.DataFrame()
new_df["text"] = df["text"]
new_df["labels"] = df["Final_label"]


class MultiLabelDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length', # New: replaces deprecated pad_to_max_length (pad_to_max_length=True,)
            truncation=True,      # explicitly set truncation
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float),
            'index': index  # Add index for tracking misclassifications
        }


# Creating the dataset and dataloader for the neural network

train_size = TRAIN_SIZE

# Convert one-hot vectors to class indices
label_indices = new_df["labels"].apply(lambda x: np.argmax(x)).tolist()

train_data, test_data = train_test_split(
    new_df, 
    test_size=1 - TRAIN_SIZE, 
    stratify=label_indices, 
    random_state=SEED
)

train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)


print("FULL Dataset: {}".format(new_df.shape))
print("TRAIN Dataset: {}".format(train_data.shape))
print("TEST Dataset: {}".format(test_data.shape))

training_set = MultiLabelDataset(train_data, tokenizer, MAX_LEN)
testing_set = MultiLabelDataset(test_data, tokenizer, MAX_LEN)



train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)


# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = BERTmodel
        self.l2 = torch.nn.Dropout(DROPOUT)
        #Change the secound val to the number of classes !!!!!!!!
        self.l3 = torch.nn.Linear(768, len(unique_labels))

    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

model = BERTClass()
model.to(device)

### LOSS, ACCURACY, OPTIMIZER

def which_loss():
    if Binary_Classification:
        return torch.nn.BCEWithLogitsLoss()  # Use BCE for binary classification

    if LOSS_FUNCTION == "weighted" or LOSS_FUNCTION == "l1+weighted":
        # Convert one-hot encoded labels to class indices, based on class distributions
        class_indices = [label.index(1) for label in train_data['labels']]
        class_counts = torch.bincount(torch.tensor(class_indices))
        class_weights = 1.0 / class_counts.float()  # inverse frequency
        class_weights = class_weights / class_weights.sum()  # normalization
        return torch.nn.CrossEntropyLoss(reduction='none', weight=class_weights.to(device))
    elif LOSS_FUNCTION == "l1":
        return torch.nn.CrossEntropyLoss(reduction='none')
    else:
        return torch.nn.CrossEntropyLoss()

CE_loss_fn = which_loss() # Sets Loss function based on the LOSS_FUNCTION and Binary_Classification variable

def loss_fn(outputs, targets, preds):
    # return torch.nn.CrossEntropyLoss()(outputs, targets) # BCEWithLogistsLoss() for Softmax, CrossEntropyLoss() for OneHot
    target_indices = torch.argmax(targets, dim=1)  # [batch_size]

    if Binary_Classification:
        return CE_loss_fn(outputs, targets)  # For binary classification, use BCEWithLogitsLoss

    if LOSS_FUNCTION == "l1" or LOSS_FUNCTION == "l1+weighted":
        # --- L1 Distance Weighting ---
        pred_indices = torch.argmax(preds, dim=1)
        target_indices = torch.argmax(targets, dim=1)
        l1_dist = torch.abs(pred_indices - target_indices).float()
        l1_weights = 1.0 + l1_dist  # Base weight=1, scaled by L1 distance

        return (CE_loss_fn(outputs, target_indices) * l1_weights).mean()  # Mean L1 weighted loss
    
    elif LOSS_FUNCTION == "weighted" or LOSS_FUNCTION == "None":
        return CE_loss_fn(outputs, target_indices).mean()  

def accuracyTest(outputs, targets):
    outputs = outputs.to(device, dtype=torch.int)
    targets = targets.to(device, dtype=torch.int)

    correct = torch.all(outputs == targets, dim=1)

    correct_num = correct.sum().item()

    accuracy = correct_num / len(targets)

    return accuracy

def l1_score(outputs, targets):
    pred_indices = torch.argmax(outputs, dim=1)
    target_indices = torch.argmax(targets, dim=1)
    l1_dists = torch.abs(pred_indices - target_indices).float()

    return l1_dists.mean().item()


optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


train_accuracies_pr_epoch = []
test_accuracies_pr_epoch = []

train_l1_scores_pr_epoch = []
test_l1_scores_pr_epoch = []

train_losses = []
confusion_matrix_pr_epoch = []
misclassified_samples = []

def train(epoch):
    model.train()
    total_loss = 0  # Track total loss for the epoch
    train_accuracies = []
    train_l1_scores = []

    for batch_idx,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)
        probs = torch.softmax(outputs, dim=1) # SOFTMAX (dim=1) or SIGMOID (), depends on how we interpret the multiclass
    
        max_indices = torch.argmax(probs, dim=1)
        preds = torch.zeros_like(probs)
        preds.scatter_(1, max_indices.unsqueeze(1), 1)

        # print(preds)
        # print(targets)

        batch_train_accuracy = accuracyTest(preds, targets)
        train_accuracies.append(batch_train_accuracy)
        print(f"Batch {batch_idx} Train Accuracy:", batch_train_accuracy)

        if not Binary_Classification:
            batch_l1_score = l1_score(preds, targets)
            train_l1_scores.append(batch_l1_score)
            print(f"Batch {batch_idx} L1 Scores:", batch_l1_score)
        
        loss = loss_fn(outputs, targets, preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Print loss every 10 batches (adjust as needed)
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')

    
    train_accuracies_pr_epoch.append(np.mean(train_accuracies)) # Save avrage accuracies for the epoch
    if not Binary_Classification:
        train_l1_scores_pr_epoch.append(np.mean(train_l1_scores)) # Save avrage L1 scores for the epoch
    avg_train_loss = total_loss / len(training_loader)
    train_losses.append(avg_train_loss)  # Save average loss for the epoch
    print(f'Epoch {epoch} Average Training Loss: {avg_train_loss}, Avrage Training Accuraccy: {np.mean(train_accuracies)}')


def validation(epoch):
    model.eval()
    confusion_matrix = np.zeros((len(unique_labels), len(unique_labels)))

    test_acc = []
    test_l1 = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            batch_indices = data['index'].numpy()
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)

            probs = torch.softmax(outputs, dim=1) # SOFTMAX (dim=1) or SIGMOID (), depends on how we interpret the multiclass
        
            max_indices = torch.argmax(probs, dim=1)
            preds = torch.zeros_like(probs)
            preds.scatter_(1, max_indices.unsqueeze(1), 1)

            print(preds)
            print(targets)

            # Print confusion matrix for each sample
            for i, (target, pred) in enumerate(zip(targets, preds)):
                true_class = target.argmax().item()
                predicted_class = pred.argmax().item()
                
                # Print per-sample prediction
                print(f"Sample {i}: true class = {true_class}, predicted = {predicted_class}")
                # Update confusion matrix
                confusion_matrix[true_class, predicted_class] += 1
                if not true_class == predicted_class:
                    orig_idx = batch_indices[i]
                    misclassified_samples.append({
                        'epoch': epoch,
                        'index': orig_idx,
                        'true_label': unique_labels[true_class],
                        'predicted_label': unique_labels[predicted_class],
                        'text': test_data.iloc[orig_idx]['text']
                    })

            

            batch_test_acc = accuracyTest(preds, targets)
            test_acc.append(batch_test_acc)
            print(f"Batch Test Accuracy: {batch_test_acc}")
            if not Binary_Classification:
                batch_test_l1 = l1_score(preds, targets)
                test_l1.append(batch_test_l1)
                print(f"Batch Test L1 Score: {batch_test_l1}")
            
    # Print confusion matrix for the epoch        
    print(confusion_matrix)
    confusion_matrix_pr_epoch.append(confusion_matrix)  # Save confusion matrix for the epoch
    # Calculate metrics
    test_accuracies_pr_epoch.append(np.mean(test_acc)) # Save avrage accuracies for the epoch
    if not Binary_Classification:
        test_l1_scores_pr_epoch.append(np.mean(test_l1)) # Save avrage L1 scores for the epoch
    print(f'Epoch {epoch}, Avrage Test Accuraccy: {np.mean(test_acc)}')


### Epoch Loop

for epoch in range(EPOCHS):
    train(epoch)
    validation(epoch)


### Plots, Text & Metrics

## Confusion Matrix
# Convert Confusion Matrix to a serializable format and add epoch info
data_to_save = {
    f"epoch_{i+1}": matrix.tolist() 
    for i, matrix in enumerate(confusion_matrix_pr_epoch)
}

# Save to JSON file
with open(jsonName, 'w') as f:
    json.dump(data_to_save, f, indent=4)

## Save misclassified samples, CSV
misclassified_df = pd.DataFrame(misclassified_samples)
misclassified_df.to_csv(misclassifiedName, index=False)


## Plots
epochs = list(range(EPOCHS))

# Accuracy Plot
plt.plot(epochs, train_accuracies_pr_epoch, label='Train Accuracy')
plt.plot(epochs, test_accuracies_pr_epoch, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Validation Accuracy")
plt.ylim(0, 1) 
plt.legend()
plt.grid(True)

plt.savefig("ACCURACY_"+graphname)


# L1 Scores Plot
if not Binary_Classification:
    plt.plot(epochs, train_l1_scores_pr_epoch, label='Train L1 Scores')
    plt.plot(epochs, test_l1_scores_pr_epoch, label='Validation L1 Scores')
    plt.xlabel("Epoch")
    plt.ylabel("L1")
    plt.title("Train vs Validation L1 Scores")
    plt.legend()
    plt.grid(True)

    plt.savefig("L1_"+graphname)







