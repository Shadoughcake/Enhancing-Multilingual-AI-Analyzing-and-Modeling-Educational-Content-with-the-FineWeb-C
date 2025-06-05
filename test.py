# Importing stock ML Libraries
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from collections import Counter
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# Sections of config

# Defining some key variables that will be used later on in the training
MAX_LEN = 128
TRAIN_SIZE = 0.05
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
EPOCHS = 10
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
        self.data =  dataframe.reset_index(drop=True)
        self.text = dataframe.text
        self.targets = self.data.labels
        self.max_len = max_len
        self.indices = list(range(len(dataframe))) 
        
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
            'index': index
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
training_set.df = train_data  # Attach dataframe for reference
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


def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets) # BCEWithLogistsLoss() for Softmax, CrossEntropyLoss() for OneHot

def accuracyTest(outputs, targets):
    outputs = outputs.to(device, dtype=torch.int)
    targets = targets.to(device, dtype=torch.int)

    correct = torch.all(outputs == targets, dim=1)

    correct_num = correct.sum().item()

    accuracy = correct_num / len(targets)

    return accuracy


optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


train_accuracies_pr_epoch = []
val_accuracies_pr_epoch = []
train_losses = []

def train(epoch):
    model.train()
    total_loss = 0  # Track total loss for the epoch
    train_accuracies = []

    for batch_idx,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)
        targets_ce = torch.argmax(targets, dim=1).to(device, dtype=torch.long) # For Cross Entropy

        outputs = model(ids, mask, token_type_ids)
        probs = torch.softmax(outputs, dim=1) # SOFTMAX (dim=1) or SIGMOID (), depends on how we interpret the multiclass
    
        max_indices = torch.argmax(probs, dim=1)
        preds = torch.zeros_like(probs)
        preds.scatter_(1, max_indices.unsqueeze(1), 1)

        print(preds)
        print(targets)
        batch_train_accuracy = accuracyTest(preds, targets)
        train_accuracies.append(batch_train_accuracy)

        print(batch_train_accuracy)
        
        loss = loss_fn(outputs, targets_ce)
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

val_accuracies = []
val_f1_micro = []
val_f1_macro = []
test_accuracies_pr_epoch = []

def validation(epoch):
    model.eval()
    all_targets = []
    all_preds = []
    model.eval()
    misclassified_samples = []  # Store misclassified examples
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            batch_indices = data['index'].numpy()
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            targets_indices = torch.argmax(targets, dim=1).cpu().numpy()
            
            
            probs = torch.softmax(outputs, dim=1)
            max_indices = torch.argmax(probs, dim=1)
            preds = torch.zeros_like(probs)
            preds.scatter_(1, max_indices.unsqueeze(1), 1)
            preds_indices = max_indices.cpu().numpy()
            all_targets.extend(targets_indices)
            all_preds.extend(preds_indices)
            
        if epoch == EPOCHS - 1:  # Final epoch only
            print("\nClassification Report:")
            print(classification_report(
                all_targets, 
                all_preds, 
                target_names=unique_labels,
                zero_division=0
            ))
            cm = confusion_matrix(all_targets, all_preds)
            plt.figure(figsize=(10,8))
            sns.heatmap(cm, annot=True, fmt='d', 
                        xticklabels=unique_labels, 
                        yticklabels=unique_labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig("confusion_matrix.png")
            for i in range(len(targets)):
                if not torch.equal(preds[i], targets[i]):
                    orig_idx = batch_indices[i]
                    misclassified_samples.append({
                        'text': test_data.iloc[orig_idx]['text'],
                        'true_label': unique_labels[torch.argmax(targets[i]).item()],
                        'predicted_label': unique_labels[torch.argmax(preds[i]).item()],
                        'original_labels': test_data.iloc[orig_idx]['educational_value_labels']
                    })
    
    # Save misclassified samples
    mis_df = pd.DataFrame(misclassified_samples)
    mis_df.to_csv(f"misclassified_epoch_{epoch}.csv", index=False)
    return mis_df
### Epoch Loop

for epoch in range(EPOCHS):
    train(epoch)
    validation(epoch)


### Plots
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