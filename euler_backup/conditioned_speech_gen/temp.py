import re
import os
import glob
import pandas as pd
import torch
import torch.optim as optim
import numpy as np
import random
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import Dataset
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, average_precision_score, roc_auc_score

os.environ["CURL_CA_BUNDLE"]=""

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
net = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")