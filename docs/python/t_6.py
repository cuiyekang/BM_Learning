import os
import pandas as pd
import numpy as np

x = np.arange(30).reshape(10,3)

x[:,1]=1

from sklearn.feature_selection import VarianceThreshold

vt = VarianceThreshold()
xt=vt.fit_transform(x)


adult_filename = "./docs/python/t_data/adult.data"
adult= pd.read_csv(adult_filename,
                    header=None,
                    names=["Age", "Work-Class", "fnlwgt", "Education",
                           "Education-Num", "Marital-Status", "Occupation",
                           "Relationship", "Race", "Sex", "Capital-gain",
                            "Capital-loss", "Hours-per-week", "Native-Country",
                            "Earnings-Raw"])

adult.dropna(how='all',inplace=True)

# print(adult['Hours-per-week'].describe())
# print(adult['Education-Num'].median())
# print(adult['Work-Class'].unique())

x=adult[['Age','Education-Num','Capital-gain','Capital-loss','Hours-per-week']].values
y=(adult['Earnings-Raw']==' >50K').values

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

transformer = SelectKBest(score_func = chi2,k=3)

xt_chi2=transformer.fit_transform(x,y)

print(transformer.scores_)

from scipy.stats import pearsonr

def multivariate_pearsonr(x,y):
    scores,pvalues=[],[]
    for column in range(x.shape[1]):
        cur_score,cur_p = pearsonr(x[:,column],y)
        scores.append(abs(cur_score))
        pvalues.append(cur_p)
    return (np.array(scores),np.array(pvalues))

transformer = SelectKBest(score_func=multivariate_pearsonr,k=3)
xt_pearson = transformer.fit_transform(x,y)

print(transformer.scores_)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

clf=DecisionTreeClassifier(random_state=14)
scores_chi2 = cross_val_score(clf,xt_chi2,y,scoring='accuracy')
scores_pearson = cross_val_score(clf,xt_pearson,y,scoring='accuracy')

print("Chi2 performance: {0:.3f}".format(scores_chi2.mean()))
print("Pearson performance: {0:.3f}".format(scores_pearson.mean()))



