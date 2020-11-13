import os
import numpy as np
import pandas as pd

data_filename = "./docs/python/t_data/ad.data"

def convert_number(x):
    try:
        return float(x)
    except ValueError:
        return np.nan

from collections import defaultdict

converters = defaultdict(convert_number)
for i in range(1558):
    converters[i]=convert_number

converters[1558] = lambda x:1 if x.strip() == "ad." else 0

ads = pd.read_csv(data_filename,header=None,converters=converters)
#ads = ads.applymap(lambda x : np.nan if isinstance(x,str) and not x=='ad.' else x)

print(ads[:5])

#ads[[0,1,2]]=ads[[0,1,2]].astype(float)

ads = ads.astype(float).dropna()
x=ads.drop(1558,axis=1).values
y=ads[1558]

print(x.shape)
print(y.shape)


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

clf=DecisionTreeClassifier(random_state=14)
scores=cross_val_score(clf,x,y,scoring='accuracy')

print("The average score is {:.4f}".format(np.mean(scores)))

from sklearn.decomposition import PCA

pca=PCA(n_components=5)
xd=pca.fit_transform(x)

np.set_printoptions(precision=3,suppress=True)
print(pca.explained_variance_ratio_)
print(pca.components_[0])

clf=DecisionTreeClassifier(random_state=14)
scores_reduced = cross_val_score(clf,xd,y,scoring="accuracy")

print("The average score from the reduced dataset is {:.4f}".format(np.mean(scores_reduced)))

from matplotlib import pyplot as plt

classes = set(y)
colors=['red','green']

for cur_class,color in zip(classes,colors):
    mask = (y==cur_class).values
    plt.scatter(xd[mask,0],xd[mask,1],marker='o',color=color,label=int(cur_class))

plt.legend()
plt.show()


