import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model



data = pd.read_csv("student-mat.csv", sep=";")
#print(data.head())
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
#print(data.head())

predict = "G3"

X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y ,test_size=0.1)

linear = linear_model.LinearRegression()  #model treningowy do wyznaczenia regresji

linear.fit(x_train, y_train)  # model dopasowuje najlepsze dane

acc = linear.score(x_test,y_test) # zwraca wartość, która repreztnuje dokładność modelu
print(acc)

print("Coefficient: \n", linear.coef_) # da liste wspołczynników nachylenia
# wpółczynników jest 5 bo linia jest w przestrzeni 5 - wymiarowej
print("Intercept: \n", linear.intercept_) # da przecięcia z osią OY

predictions = linear.predict(x_test)  # przypisanie do zminnej predictions wszystkich prognóz

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x]) # y_test to końcowa ocena

