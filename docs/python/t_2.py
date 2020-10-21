from sklearn.datasets import load_iris
import numpy as np

dataset = load_iris()
x=dataset.data
y=dataset.target

n_samples,n_features=x.shape

attrabute_means=x.mean(axis=0)

x_d=np.array(x >= attrabute_means,dtype='int')

from sklearn.model_selection import train_test_split

random_state = 14
x_train,x_test,y_train,y_test=train_test_split(x_d,y,random_state=random_state)

print("There are {} training samples".format(y_train.shape))
print("There are {} testing samples".format(y_test.shape))

from collections import defaultdict
from operator import itemgetter



