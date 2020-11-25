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





