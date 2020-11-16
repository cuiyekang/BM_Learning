from collections import Counter

s = "Three Rings for the Elven-kings under the sky,Seven for the Dwarf-lords in halls of stone,Nine for Mortal Men,"
s += "doomed to die,One for the Dark Lord on his dark throneIn the Land of Mordor where the Shadows lie."
s += "One Ring to rule them all, One Ring to find them,"
s += "One Ring to bring them all and in the darkness bind them."
s += "In the Land of Mordor where the Shadows lie. "
s= s.lower()

words = s.split()

c = Counter(words)

#print(c.most_common(5))

import os
import json

input_filename = "./docs/python/t_data/python_tweets.json"
classes_filename="./docs/python/t_data/python_classes.json"

tweets =[]

with open(input_filename) as inf:
    for line in inf:
        if len(line.strip())==0:
            continue
        tweets.append(json.loads(line)['text'])

#print("Loaded {} tweets".format(len(tweets)))

with open(classes_filename) as inf:
    labels = json.load(inf)

n_samples=min(len(tweets),len(labels))

sample_tweets = [t.lower() for t in tweets[:n_samples]]
labels = labels[:n_samples]


import numpy as np

y_true =np.array(labels)
print("{:.1f}% have class 1".format(np.mean(y_true == 1) * 100))

from sklearn.base import TransformerMixin
from nltk import word_tokenize

class NLTKBOW(TransformerMixin):
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        return [{word: True for word in word_tokenize(document)}
                for document in X]

from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

#print(word_tokenize("long time no see,I miss you very much"))


pipeline=Pipeline([('big-of-words',NLTKBOW()),
                    ('vectorizer',DictVectorizer()),
                    ('naive-bayes',BernoulliNB())])

scores = cross_val_score(pipeline,sample_tweets,y_true,cv = 10,scoring='f1')

print("Score: {:.3f}".format(np.mean(scores)))

model = pipeline.fit(tweets,labels)
nb=model.named_steps["naive-bayes"]
feature_probabilities = nb.feature_log_prob_

top_features = np.argsort(-feature_probabilities[1])[:50]
dv = model.named_steps["vectorizer"]

for i,feature_index in enumerate(top_features):
    print(i,dv.feature_names_[feature_index],np.exp(feature_probabilities[1][feature_index]))

