import numpy as np
import torch
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression
import importlib
import lib.processing.data_processing as lib
from sklearn.metrics import confusion_matrix

importlib.reload(lib)

#read the data
TRAIN_FP = 'data/bias_data/WNC/biased.word.train'
TEST_FP = 'data/bias_data/WNC/biased.word.test'

wnc_train = lib.raw_data(TRAIN_FP, 3, 4)
wnc_train_df = wnc_train.add_miss_word_col(dtype='df')

wnc_test = lib.raw_data(TEST_FP, 3, 4)
wnc_test_df = wnc_test.add_miss_word_col(dtype='df')

train_data_num=1000
train_df = wnc_train.make_training_array(wnc_train_df)
train_df=train_df.head(train_data_num)
test_data_num=200
test_df = wnc_test.make_training_array(wnc_test_df)
test_df=test_df.head(test_data_num)

model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

tokenized_train = train_df["string"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
tokenized_test = test_df["string"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

#padding
length_of_tokens_train=np.empty(tokenized_train.shape[0])
for i in range(tokenized_train.shape[0]):
    length_of_tokens_train[i]=len(tokenized_train[i])
padding_length_train=max(length_of_tokens_train)
for i in range(tokenized_train.shape[0]):
    tokenized_train[i]+=[0 for i in range(int(padding_length_train)-len(tokenized_train[i]))]

length_of_tokens_test=np.empty(tokenized_test.shape[0])
for i in range(tokenized_test.shape[0]):
    length_of_tokens_test[i]=len(tokenized_test[i])
padding_length_test=max(length_of_tokens_test)
for i in range(tokenized_test.shape[0]):
    tokenized_test[i]+=[0 for i in range(int(padding_length_test)-len(tokenized_test[i]))]

input_ids_train = torch.tensor(tokenized_train)
input_ids_test = torch.tensor(tokenized_test)

with torch.no_grad():
    last_hidden_states_train = model(input_ids_train)
with torch.no_grad():
    last_hidden_states_test = model(input_ids_test)

# Slice the output for the first position for all the sequences, take all hidden unit outputs
features_train = last_hidden_states_train[0][:,0,:].numpy()
features_test = last_hidden_states_test[0][:,0,:].numpy()

labels_train = train_df["label"]
labels_test = test_df["label"]

lr_clf = LogisticRegression(max_iter=5000)
lr_clf.fit(features_train, labels_train)

y_pred=lr_clf.predict(features_test)
Confusion_Matrix=confusion_matrix(labels_test,y_pred)

print(Confusion_Matrix)