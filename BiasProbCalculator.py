import transformers
import torch
import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lib.data_processing as lib
import importlib
from collections import defaultdict
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import transformers as tf # pytorch transformers
import math


class WNCDataset(Dataset):
    def __init__(self, sentences, targets, tokenizer, max_len):
        self.sentences = sentences
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = str(self.sentences[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'sentence': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = WNCDataset(
    sentences=df.string.to_numpy(),
    targets=df.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len,
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )

def train_probability_calculator(train_data):

    weights = training(train_data, 0.005, 20)
    return weights


def probability(input, weights):
    res = weights[0]  # bias
    for i in range(1, len(input)):
        res += input[i - 1] * weights[i]

    return res


def prediction(value):  # modify the return value of the prediction according to the labels.
    activation = sigmoid(value)
    if activation >= 0.5:
        return 1
    else:
        return 0


def sigmoid(u):
    return (1 / (1 + math.pow(math.exp(1), -u)))



# train the model, return a vector of weights
def training(train_data, label, learning_rate, epoch):
    weights = [0.0 for i in range(len(train_data[0]))]
    for i in range(epoch):
        count_wrong = 0
        for i in range(len(train_data)):
            data = train_data[i]
            if len(data) == 0:
                break
            confidence = (probability(data, weights))
            pred = prediction(confidence)
            error = label[i] - pred
            if error != 0:
                count_wrong += 1
            weights[0] = weights[0] + learning_rate * error
            for i in range(1, len(weights)):
                weights[i] += learning_rate * error * data[i - 1]

    return weights

#
# dataset format: raw sentences and labels.
#
def train_calculator(sentences, label):

    # print(features)
    features = get_bert_features(sentences)

    weights = training(features, label, 0.0005, 120)

    return weights

#use BERT to transform sentences into vectors
def get_bert_features(sentences):
    model = tf.BertModel.from_pretrained('bert-base-uncased')

    tokenizer = tf.BertTokenizer.from_pretrained('bert-base-uncased')

    tokenized = sentences.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    # print(tokenized.head())

    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
    mask = np.where(padded != 0, 1, 0)
    input_ids = torch.tensor(padded)
    mask = torch.tensor(mask)

    with torch.no_grad():
        last_hidden_states = model(input_ids=input_ids, attention_mask=mask)

    features = last_hidden_states[0][:, 0, :].numpy()

    return features

#calculate the bias score for a sentence
def get_bias_probability(sentences, calculator):
    features = get_bert_features(sentences)
    res = list()
    for i in range(len(features)):
        res.append(probability(features[i], calculator))
    return res


def main():
    importlib.reload(lib)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    TRAIN_FP = '/Users/mac/bias-scoring-algorithm/bias_data/WNC/biased.word.train'
    TEST_FP = '/Users/mac/bias-scoring-algorithm/bias_data/WNC/biased.word.test'

    wnc_train = lib.raw_data(TRAIN_FP, 3, 4)
    wnc_train_df = wnc_train.add_miss_word_col(dtype='df')

    wnc_test = lib.raw_data(TEST_FP, 3, 4)
    wnc_test_df = wnc_test.add_miss_word_col(dtype='df')
    #
    train_df = wnc_train.make_training_array(wnc_train_df)
    test_df = wnc_test.make_training_array(wnc_test_df)

    train_df = train_df[:20]
    test_df = test_df[:20]

    label_train = train_df['label']
    sentences_train = train_df['string']

    label_test = test_df['label']
    sentences_test = test_df['string']
    #print(sentences_train.head())
    # print(train_df.head())
    # print(test_df.head())

    calculator = train_calculator(sentences_train, label_train)

    prob = get_bias_probability(sentences_test,calculator)

    print(prob)

    print(label_test)

if __name__ == '__main__':
    main()