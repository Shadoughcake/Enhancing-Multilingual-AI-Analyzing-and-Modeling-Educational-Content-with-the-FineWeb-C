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
 

DATASET = pd.read_csv("Enhancing-Multilingual-AI-Analyzing-and-Modeling-Educational-Content-with-the-FineWeb-C/annotations Data/fineweb-c_relabled.csv")

print(DATASET)