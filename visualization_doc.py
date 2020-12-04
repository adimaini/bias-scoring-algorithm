import lib.data_processing as lib
import importlib
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import texthero as hero

df = pd.read_csv('./bias_data/bias_data/transcripts/transcripts.csv')
print(df.shape)


df = df.iloc[:1000, :]
df['clean_text'] = hero.clean(df['transcript'])
df['tfidf_clean_text'] = hero.tfidf(df['clean_text'])
df['pca'] = hero.pca(df['tfidf_clean_text'], 3)

print(df.head(5))

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
x = list(x[0] for x in df['pca'])
y = list(x[1] for x in df['pca']) 
z = list(x[2] for x in df['pca'])
sc = ax.scatter(x, y, z)
plt.show()
plt.colorbar(sc)