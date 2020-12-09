import lib.processing.data_processing as lib
import importlib
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import texthero as hero
from collections import Counter

df = pd.read_csv('./data/bias_data/transcripts/transcripts.csv')
print(df.shape)


df = df.iloc[:2000, :]
df['clean_text'] = hero.clean(df['transcript'])
df['tfidf_clean_text'] = hero.tfidf(df['clean_text'], max_features=200)
df['pca'] = hero.pca(df['tfidf_clean_text'], 3)

# print(df.head(5))
print(Counter(list(df['host'])).keys())
print(Counter(list(df['host'])).values())
# print(list(df['clean_text'])[:100])
hostNum = len(Counter(list(df['host'])).values())

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
x = list(x[0] for x in df['pca'])
y = list(x[1] for x in df['pca']) 
z = list(x[2] for x in df['pca'])
maxLen = max([len(x) for x in df['host']])
c = []
nameMapping = dict()
for i in df['host']:
    c.append(len(i)/maxLen)
    nameMapping[len(i)/maxLen] = i

fig.suptitle('Transcript Dataset Visualization (2000 samples)')
sc = ax.scatter(x, y, z, c=c, cmap=plt.cm.get_cmap('Accent',hostNum))
formatter = plt.FuncFormatter(lambda val, loc: nameMapping[val])

plt.colorbar(sc, ticks=list(Counter(c).keys()), format=formatter)
plt.show()
