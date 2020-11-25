import numpy as np
import csv
import os

# home_folder = os.path.expanduser("~")
# print(home_folder)

data_filename=os.path.join('.','docs','python','t_data','ionosphere.data')

x = np.zeros((351,34),dtype='float')
y = np.zeros(351,dtype='bool')

with open(data_filename,'r') as input_file:
    reader = csv.reader(input_file)
    for i,row in enumerate(reader):
        data=[float(datum) for datum in row[:-1]]
        x[i]=data
        y[i]=row[-1]=='g'

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=14)

print("There are {} samples in the training dataset".format(x_train.shape[0]))
print("There are {} samples in the testing dataset".format(x_test.shape[0]))
print("Each sample has {} features".format(x_train.shape[1]))

from sklearn.neighbors import KNeighborsClassifier

estimator=KNeighborsClassifier()
estimator.fit(x_train,y_train)

y_predicted=estimator.predict(x_test)

accuracy = np.mean(y_test == y_predicted)*100

print("The accuracy is {0:.1f}%".format(accuracy))


from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator,x,y,scoring='accuracy')

average_accuracy = np.mean(scores)*100

print("The average accuracy is {0:.1f}%".format(average_accuracy))


# avg_scores = []
# all_scores = []

# paramter_values = list(range(1,21))
# for n_neighbors in paramter_values:
#     estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
#     scores=cross_val_score(estimator,x,y,scoring='accuracy')
#     avg_scores.append(np.mean(scores))
#     all_scores.append(scores)

from matplotlib import pyplot as plt

# plt.figure(figsize = (32,20))
# plt.plot(paramter_values,avg_scores,'-o',linewidth=5,markersize=24)

# plt.show()


x_broken =np.array(x)

x_broken[:,::2] /= 10

estimator = KNeighborsClassifier()
original_scores = cross_val_score(estimator,x,y,scoring='accuracy')
original_scores_mean = np.mean(original_scores)*100
print("The original accuracy is {0:.1f}%".format(original_scores_mean))

broken_scores = cross_val_score(estimator,x_broken,y,scoring='accuracy')
broken_scores_mean=np.mean(broken_scores)*100
print("The broken accuracy is {0:.1f}%".format(broken_scores_mean))

from sklearn.preprocessing import MinMaxScaler

x_transformed = MinMaxScaler().fit_transform(x_broken)

transformed_scores = cross_val_score(estimator, x_transformed,y,scoring='accuracy')
transformed_scores_mean = np.mean(transformed_scores)*100
print("The transformed accuracy is {0:.1f}%".format(transformed_scores_mean))


from sklearn.pipeline import Pipeline

scaling_pipeline = Pipeline([('scale',MinMaxScaler()),
                            ('predict',KNeighborsClassifier())])


pipeline_scores = cross_val_score(scaling_pipeline,x_broken,y,scoring='accuracy')
pipeline_scores_mean = np.mean(pipeline_scores)*100
print("The pipeline accuracy is {0:.1f}%".format(pipeline_scores_mean))



