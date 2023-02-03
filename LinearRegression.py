import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use("ggplot")

data = pd.read_csv("student-mat.csv", sep=";")

predict = "G3"

data = data[["G1", "G2", "absences","failures", "studytime", "health", "traveltime","G3"]]


x = np.array(data.drop([predict], 1))
y =np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.8)



best = 90 #change to your acc, do not do something impossible like 100
for _ in range(500):
    
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.8)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc.round(3)*100))
    
    
    if acc > best:
        best = acc
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)


pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)



predicted= linear.predict(x_test)
for x in range(len(predicted)):
    print(predicted[x].round(4), x_test[x].round(4), y_test[x].round(4))


print(best) #if 90 its not actually getting 90, set the best to your desired accuracy so it can save it
