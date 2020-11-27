#N元词方法处理作者归属问题

import sklearn
#print(sklearn.__version__)
import matplotlib.pyplot as plt
import numpy as np

#二元的Bi-Gram和三元的Tri-Gram

n = 3  # Length of n-gram
suess = "Sam! If you will let me be, I will try them. You will see."
for i in range(5):  # Print the first five n-grams
    print("n-gram #{}: {}".format(i, suess[i:i+n]))

from collections import defaultdict

def count_ngrams(document,n,normalise=False):
    "将文档拆成n元词键值，词为键，数量为值"
    counts = defaultdict(float)
    for i in range(len(document) - n + 1):
        ngram=document[i:i+n]
        counts[ngram]= counts[ngram]+1
    if normalise:
        normalise_factor = float(len(document) - n + 1)
        for ngram in counts:
            counts[ngram] = counts[ngram] / normalise_factor
    return counts

counts = count_ngrams(suess,3)
for ngram in [" I ", "Sam", "ill", "abc"]:
    print("{0} occurs {1:.0f} time(s)".format(ngram, counts[ngram]))

from operator import itemgetter
top_ngrams = sorted(counts.items(),key=itemgetter(1),reverse=True)[:10]
ngrams,ng_counts=zip(*top_ngrams)
y_pos = np.arange(len(ngrams))

# plt.figure()
# plt.barh(y_pos,ng_counts,align='center',alpha=0.4)
# plt.yticks(y_pos,ngrams)
# plt.xlabel('Count')
# plt.title('Most frequent n-grams')
# plt.xticks(np.arange(0,max(ng_counts)+1,1.0))
# plt.show()

frequencies = count_ngrams(suess,3,normalise = True)
for ngram in [" I ", "Sam", "ill", "abc"]:
    print("{0} has frequency {1:.3f}".format(ngram, frequencies[ngram]))

import os
import codecs

default_folder = "./docs/python/course2/t_data/books"

def get_single_corpus(folder = None):
    documents=[]
    authors=[]
    training_mask=[]
    authornum=0
    i=0
    subfolders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder,name))]
    for subfolder in subfolders:
        sf = os.path.join(folder,subfolder)
        #print("Author %d is %s" % (authornum, subfolder))
        for document_name in os.listdir(sf):
            i+=1
            with codecs.open(os.path.join(sf,document_name),encoding='utf=8') as input_f:
                text = cleanFile(input_f.read())
                #获取到数据中，每个文件夹下只有一个文件，这里简单处理，将这个文件分成10份，当成10个文件
                every = int(len(text)/10)
                for j in np.arange(10):
                    if (j+1)*every > len(text):
                        documents.append(text[j*every:(j+1)*every])
                    else:
                        documents.append(text[j*every:len(text)])
                    authors.append(authornum)
                    training_mask.append(True)
        authornum += 1
    # min_docs = 10
    # c = np.bincount(authors)
    # print(c)
    return documents,np.array(authors,dtype='int')

def cleanFile(document):
    lines = document.split("\n")
    start = 0
    end = len(lines)
    for i in range(len(lines)):
        line = lines[i]
        if line.startswith("*** START OF THIS PROJECT GUTENBERG"):
            start = i + 1
        elif line.startswith("*** END OF THIS PROJECT GUTENBERG"):
            end = i - 1
    return "\n".join(lines[start:end])



documents,classes = get_single_corpus(default_folder)
documents=np.array(documents,dtype='object')

# print(documents[0].split("\n")[:20])

# counts = count_ngrams(documents[0], 3, normalise=True)
# top_ngrams = sorted(counts.items(), key=itemgetter(1), reverse=True)[:30]
# ngrams, ng_counts = zip(*top_ngrams)
# x_pos = np.arange(len(ngrams))

# plt.figure()
# plt.bar(x_pos, ng_counts, alpha=0.4)
# plt.xticks(x_pos,ngrams,rotation=60)
# plt.xlabel('n-gram (by rank)')
# plt.ylabel('Normalised Frequency')
# plt.title('Frequency of the most common n-grams')
# plt.xlim(0, len(ngrams))
# plt.show()

from sklearn.base import BaseEstimator,ClassifierMixin

class SCAP(BaseEstimator,ClassifierMixin):
    def __init__(self,n,L):
        self.n=n
        self.L=L
        self.author_profiles=None #每个作者的前L个关键字
    
    def create_profile(self,documents):
        #如果是单字符串，转为字符串组
        if isinstance(documents,str):
            documents = [documents,]
        profiles = (count_ngrams(document,self.n,normalise=False) for document in documents)

        main_profile = defaultdict(float)
        for profile in profiles:
            for ngram in profile:
                main_profile[ngram] += profile[ngram]
            
        num_ngrams = sum(main_profile.values())
        for ngram in main_profile:
            main_profile[ngram] /= num_ngrams
        
        return self.top_L(main_profile)

    def top_L(self,profile):
        if self.L >= len(profile):
            return profile

        threshold = sorted(profile.values())[-self.L]
        return {ngram: profile[ngram] for ngram in profile if profile[ngram] >= threshold}
        
    def compare_profiles(self,profile1,profile2):
        "计算两组词相同的比例，然后计算距离，1-比例"
        similarity = len(set(profile1.keys()) & set(profile2.keys())) / float(self.L)
        similarity = min(similarity,1.0)
        distance = 1 - similarity
        return distance

    def fit(self,documents,classes):
        author_documents = ((author,[documents[i] for i in range(len(documents))
                                    if classes[i]==author])
                            for author in set(classes))
        
        self.author_profiles = {author:self.create_profile(cur_docs)
                                for author,cur_docs in author_documents}

    def predict(self,documents):
        predictions = np.array([self.predict_single(document) for document in documents])
        return predictions

    def predict_single(self,document):
        profile = self.create_profile(document)
        distances = [(author,self.compare_profiles(profile,self.author_profiles[author]))
                    for author in self.author_profiles]

        prediction = sorted(distances,key=itemgetter(1))[0][0]

        return prediction


model = SCAP(n=4,L=2)

# model.fit(documents,classes)
# y_pred = model.predict(documents)
# print(y_pred)
# print("Accuracy is {:.1f}%".format(100. * np.mean(classes == y_pred)))

#跑不出来，不知道为什么
from sklearn import model_selection
cv = model_selection.KFold(len(documents),n_splits=5,shuffle=True,random_state=14)
scores =model_selection.cross_val_score(model,documents,classes,cv=cv)
print("Accuracy is {:.1f}%".format(100. * np.mean(scores)))


