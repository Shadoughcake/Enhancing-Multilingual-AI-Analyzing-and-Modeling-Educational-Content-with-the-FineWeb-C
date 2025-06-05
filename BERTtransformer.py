# Importing stock ML Libraries
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from sklearn.model_selection import train_test_split
from collections import Counter
import ast
import matplotlib.pyplot as plt
import json



# Sections of config

# Defining some key variables that will be used later on in the training
MAX_LEN = 128
TRAIN_SIZE = 0.8
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



# # Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


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
DATASET = pd.read_parquet("hf://datasets/data-is-better-together/fineweb-c/dan_Latn/train-00000-of-00001.parquet")
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
    "None": unique_labels.index("None"),
    "Minimal": unique_labels.index("Minimal"),
    "Basic": unique_labels.index("Basic"),
    "Good": unique_labels.index("Good"),
    "Excellent": unique_labels.index("Excellent"),
}

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
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


# Creating the dataset and dataloader for the neural network

train_size = TRAIN_SIZE
train_data=new_df.sample(frac=train_size,random_state=200)
test_data=new_df.drop(train_data.index).reset_index(drop=True)
train_data = train_data.reset_index(drop=True)


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
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
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

# Convert one-hot encoded labels to class indices, based on class distributions
class_indices = [label.index(1) for label in train_data['labels']]
class_counts = torch.bincount(torch.tensor(class_indices))
class_weights = 1.0 / class_counts.float()  # inverse frequency
class_weights = class_weights / class_weights.sum()  # normalization

# Define loss_fn with weights
weighted_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))


def loss_fn(outputs, targets):
    # return torch.nn.CrossEntropyLoss()(outputs, targets) # BCEWithLogistsLoss() for Softmax, CrossEntropyLoss() for OneHot

    target_indices = torch.argmax(targets, dim=1)  # [batch_size]
    return weighted_loss_fn(outputs, target_indices)

def accuracyTest(outputs, targets):
    outputs = outputs.to(device, dtype=torch.int)
    targets = targets.to(device, dtype=torch.int)

    correct = torch.all(outputs == targets, dim=1)

    correct_num = correct.sum().item()

    accuracy = correct_num / len(targets)

    return accuracy


optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


train_accuracies_pr_epoch = []
test_accuracies_pr_epoch = []
train_losses = []
confusion_matrix_pr_epoch = []

def train(epoch):
    model.train()
    total_loss = 0  # Track total loss for the epoch
    train_accuracies = []

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
        
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Print loss every 10 batches (adjust as needed)
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')

    
    train_accuracies_pr_epoch.append(np.mean(train_accuracies)) # Save avrage accuracies for the epoch
    avg_train_loss = total_loss / len(training_loader)
    train_losses.append(avg_train_loss)  # Save average loss for the epoch
    print(f'Epoch {epoch} Average Training Loss: {avg_train_loss}, Avrage Training Accuraccy: {np.mean(train_accuracies)}')


def validation(epoch):
    model.eval()
    confusion_matrix = np.zeros((len(unique_labels), len(unique_labels)))

    test_acc = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
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
            

            batch_test_acc = accuracyTest(preds, targets)
            test_acc.append(batch_test_acc)
            
    # Print confusion matrix for the epoch        
    print(confusion_matrix)
    confusion_matrix_pr_epoch.append(confusion_matrix)  # Save confusion matrix for the epoch
    # Calculate metrics
    test_accuracies_pr_epoch.append(np.mean(test_acc)) # Save avrage accuracies for the epoch
    print(f'Epoch {epoch}, Avrage Test Accuraccy: {np.mean(test_acc)}')


### Epoch Loop

for epoch in range(EPOCHS):
    train(epoch)
    validation(epoch)


### Metric Functions
def tp(multi_class_confusion_matrix):
    return np.diag(multi_class_confusion_matrix)

def fp(multi_class_confusion_matrix):
    return np.sum(multi_class_confusion_matrix, axis=0) - tp(multi_class_confusion_matrix)

def fn(multi_class_confusion_matrix):
    return np.sum(multi_class_confusion_matrix, axis=1) - tp(multi_class_confusion_matrix)

def tn(multi_class_confusion_matrix):
    total = np.sum(multi_class_confusion_matrix)
    return total - (tp(multi_class_confusion_matrix) + fp(multi_class_confusion_matrix) + fn(multi_class_confusion_matrix))

def precision(tp,fp):
    return tp / (tp + fp) if (tp + fp).all() else np.nan

def recall(tp,fn):
    return tp / (tp + fn) if (tp + fn).all() else np.nan

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall).all() else np.nan

def metrics(cm):
    tp_values = tp(cm)
    fp_values = fp(cm)
    fn_values = fn(cm)
    tn_values = tn(cm)

    class_metrics = {}

    for i in range(len(cm)):
        precision_values = precision(tp_values[i], fp_values[i])
        recall_values = recall(tp_values[i], fn_values[i])
        f1_values = f1_score(precision_values, recall_values)

        class_metrics[i] = {
            "TP": tp_values[i],
            "FP": fp_values[i],
            "FN": fn_values[i],
            "TN": tn_values[i],
            "Precision": precision_values,
            "Recall": recall_values,
            "F1 Score": f1_values
        }

    return class_metrics


### Plots & Metrics

# Metrics

all_epoch_metrics = []
for epoch, cm in enumerate(confusion_matrix_pr_epoch):  # confusion_matrices = list of matrices per epoch
    class_metrics = metrics(cm)
    
    # Attach epoch info
    epoch_entry = {
        "epoch": epoch,
        "metrics": class_metrics  # class_metrics is already in {class_id: {...}} format
    }
    
    all_epoch_metrics.append(epoch_entry)

with open("epoch_metrics.json", "w") as f:
    json.dump(all_epoch_metrics, f, indent=2)


# Plots

epochs = list(range(EPOCHS))

plt.plot(epochs, train_accuracies_pr_epoch, label='Train Accuracy')
plt.plot(epochs, test_accuracies_pr_epoch, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Validation Accuracy")
plt.legend()
plt.grid(True)

plt.savefig("learningcurve.png")

plt.show()