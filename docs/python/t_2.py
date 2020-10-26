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


def train(x,y_true,feature):
    n_samples,n_features = x.shape
    
    values=set(x[:,feature])

    predictors = dict()
    errors=[]

    for current_value in values:
        most_frequent_class,error = train_feature_value(x,y_true,feature,current_value)
        predictors[current_value]=most_frequent_class
        errors.append(error)
    
    total_error=sum(errors)
    return predictors,total_error


def train_feature_value(x, y_true, feature, value):
    class_counts=defaultdict(int)
    for sample,y in zip(x,y_true):
        if sample[feature]==value:
            class_counts[y] += 1
        
    sorted_class_counts = sorted(class_counts.items(),key=itemgetter(1),reverse=True)
    most_frequent_class = sorted_class_counts[0][0]

    error = sum([class_count for class_value,class_count in class_counts.items() if class_value!=most_frequent_class])

    return most_frequent_class,error


all_predictors={variable : train(x_train,y_train,variable) for variable in range(x_train.shape[1])}


errors = {variable : error for variable,(mapping,error) in all_predictors.items()}

best_variable, best_error = sorted(errors.items(), key=itemgetter(1))[0]

print("The best model is based on variable {0} and has error {1:.2f}".format(best_variable, best_error))

model = {"vaviable":best_variable,"predictor":all_predictors[best_variable][0]}

print(model)

def predict(x_test,model):
    variable=model["vaviable"]
    predictor = model["predictor"]
    y_predicted = np.array([predictor[int(sample[variable])] for sample in x_test])
    return y_predicted

y_predicted = predict(x_test,model)

print(y_predicted)

accuracy = np.mean(y_predicted==y_test)*100

print("The test accuracy is {:.1f}%".format(accuracy))

from sklearn.metrics import classification_report

print(classification_report(y_test,y_predicted))
