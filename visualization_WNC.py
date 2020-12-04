import lib.data_processing as lib
import importlib
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import texthero as hero

importlib.reload(lib)

PATH = './bias_data/bias_data/WNC/biased.word.train' 
wnc_train = lib.raw_data(PATH, 3, 4)
wnc_train_df = wnc_train.add_miss_word_col(dtype='df')
wnc_train_df.head(5)
train_df = wnc_train.make_training_array(wnc_train_df)


df = train_df.iloc[:1000, :]
df['clean_text'] = hero.clean(df['string'])
df['tfidf_clean_text'] = hero.tfidf(df['clean_text'])
df['pca'] = hero.pca(df['tfidf_clean_text'], 3)

print(df.head(5))

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
x = list(x[0] for x in df['pca'])
y = list(x[1] for x in df['pca']) 
z = list(x[2] for x in df['pca'])
sc = ax.scatter(x, y, z, c=df['label'])
plt.show()
plt.colorbar(sc)