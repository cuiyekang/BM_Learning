#并行计算相关练习


a = [[1,2,1],[3,2],[4,9,1,0,2]]
sums = map(sum,a)
sums = []
for sublist in a:
    results = sum(sublist)
    sums.append(results)

#print(sums)

def add(a,b):
    return a + b

from functools import reduce

#print(reduce(add,sums,0))

initial = 0
current_result = initial
for element in sums:
    current_result = add(current_result,element)

#print(current_result)

from collections import defaultdict

def map_word_count(document_id,document):
    counts = defaultdict(int)
    for word in document.split():
        counts[word] += 1
    for word in counts:
        yield (word,counts[word])

def shuffle_words(results_generators):
    records = defaultdict(list)
    for results in results_generators:
        for word,count in results:
            records[word].append(count)
    for word in records:
        yield (word,records[word])
    
def reduce_counts(word,list_of_counts):
    return (word,sum(list_of_counts))

#数据下载太慢可以将文件下载后复制到 Lib\site-packages\sklearn\datasets\data
#下载的文件 20news-bydate.tar.gz 需要解压
from sklearn.datasets import fetch_20newsgroups

dataset = fetch_20newsgroups(subset = 'train')
documents = dataset.data[:50]

map_results = map(map_word_count,range(len(documents)),documents)

shuffle_results = shuffle_words(map_results)

reduce_results = [reduce_counts(word,list_of_counts) for word,list_of_counts in shuffle_results]

# print(reduce_results[:5])
# print(len(reduce_results))

from joblib import Parallel,delayed

def map_word_count_list(document_id,document):
    counts = defaultdict(int)
    for word in document.split():
        counts[word]+=1
    return list(counts.items())

map_results = Parallel(n_jobs=2)(delayed(map_word_count_list)(i,document) for i,document in enumerate(documents))

shuffle_results = shuffle_words(map_results)

print(list(shuffle_results)[:10]) 


