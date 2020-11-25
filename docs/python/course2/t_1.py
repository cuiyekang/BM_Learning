import numpy as np

dataset_filename="./docs/python/course2/t_data/affinity_dataset.txt"

x = np.loadtxt(dataset_filename)

n_samples, n_features = x.shape
#print("This dataset has {0} samples and {1} features".format(n_samples, n_features))

features = ["bread", "milk", "cheese", "apples", "bananas"]

num_apple_purchases=0
rule_valid=0
rule_invalid=0
for sample in x:
    if sample[3]==1:
        num_apple_purchases += 1
        if sample[4]==1:
            rule_valid +=1
        else:
            rule_invalid+=1

#print("{0} num_apple_purchases".format(num_apple_purchases))
#print("{0} rule_valid".format(rule_valid))
#print("{0} rule_invalid".format(rule_invalid))


support=rule_valid
confidence= rule_valid / num_apple_purchases

#print("support is {0} , confidence is {1:.3f}".format(support,confidence))
#print("As a percentage, that is {0:.1f}%.".format(100 * confidence))

from collections import defaultdict

valid_rules = defaultdict(int)
invalid_rules = defaultdict(int)
num_occurences=defaultdict(int)

for sample in x:
    for premise in range(n_features):
        if sample[premise]==0: continue
        num_occurences[premise] +=1
        for conclusion in range(n_features):
            if premise == conclusion : continue
            if sample[conclusion]==1:
                valid_rules[(premise,conclusion)] += 1
            else:
                invalid_rules[(premise,conclusion)] += 1

support=valid_rules
confidence = defaultdict(float)

for premise,conclusion in valid_rules.keys():
    confidence[(premise,conclusion)]=valid_rules[(premise,conclusion)] / num_occurences[premise]
    
#for premise, conclusion in confidence:
#    print_rule(premise,conclusion,support,confidence,features)

def print_rule(premise, conclusion, support, confidence, features):
    premise_name = features[premise]
    conclusion_name = features[conclusion]
    print("Rule: If a person buys {0} they will also buy {1}".format(premise_name, conclusion_name))
    print(" - Confidence: {0:.3f}".format(confidence[(premise, conclusion)]))
    print(" - Support: {0}".format(support[(premise, conclusion)]))
    print("")

#print_rule(1, 3, support, confidence, features)

from pprint import pprint

#pprint(list(support.items()))


from operator import itemgetter

sorted_support = sorted(support.items(),key=itemgetter(1),reverse=True)

for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_support[index][0]
    print_rule(premise, conclusion, support, confidence, features)

sorted_confidence = sorted(confidence.items(), key=itemgetter(1), reverse=True)

for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    print_rule(premise, conclusion, support, confidence, features)


    