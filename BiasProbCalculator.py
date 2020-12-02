# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import torch
import transformers as tf # pytorch transformers

#multiply the weights and our input vectors to get the probabilites of bias
def net_sum(input, weights):
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
def training(self, train_data, label, learning_rate, epoch):
    weights = [0.0 for i in range(len(train_data[0]))]
    for i in range(epoch):
        count_wrong = 0
        for i in range(len(train_data)):
            data = train_data[i]
            if len(data) == 0:
                break
            confidence = (net_sum(data, weights))
            pred = prediction(confidence)
            error = label[i] - pred
            if error != 0:
                count_wrong += 1
            weights[0] = weights[0] + learning_rate * error
            for i in range(1, len(weights)):
                weights[i] += learning_rate * error * data[i - 1]

    return weights

#
# input: raw sentences and labels
#
def bias_probability(sentences, label):
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
    # print(features)

    weights = training(features, label, 0.005, 10)

    result = list()
    for i in range(len(features)):
        prob = net_sum(features[i], weights)
        result.append(prob)

    return result