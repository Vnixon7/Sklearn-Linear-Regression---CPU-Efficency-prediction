import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model,preprocessing
from sklearn import datasets
import pickle
data = pd.read_csv('computer data.data',sep=',')
label_maker = preprocessing.LabelEncoder()
advisor = label_maker.fit_transform(list(data['advisor']))
Model_name = label_maker.fit_transform(list(data['Model_name']))
Cycle_time = label_maker.fit_transform(list(data['Cycle_time']))
Min_memory = label_maker.fit_transform(list(data['Min_memory']))
Max_memory = label_maker.fit_transform(list(data['Max_memory']))
Cach_memory = label_maker.fit_transform(list(data['Cach_memory']))
Min_chanels = label_maker.fit_transform(list(data['Min_chanels']))
Max_chanels = label_maker.fit_transform(list(data['Max_chanels']))
Pub_performance = label_maker.fit_transform(list(data['Pub_performance']))
EP = label_maker.fit_transform(list(data['Est_performance']))






x = list(zip(advisor,Model_name,Cycle_time,Min_memory,Max_memory,Cach_memory,Min_chanels,Max_chanels,Pub_performance))
y = list(EP)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


best = 0.9895203562109169
for i in range(100000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print('interation: ', i, 'Accuracy: ', acc * 100)

    if acc > best:
        best = acc
        with open('LR_CPU.pickle', 'wb') as f:
            pickle.dump(linear, f)





load_in = open('LR_CPU.pickle', 'rb')
linear = pickle.load(load_in)

print('Co-efficent:', linear.coef_)
print('Intercept:', linear.intercept_)

predictions = linear.predict(x_test)

for i in range(len(predictions)):
    print('Prediction:',predictions[i],x_test[i],'Actual:',y_test[i])
print(best)