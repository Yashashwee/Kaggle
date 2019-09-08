#Dependencies
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#load data
dataset = pd.read_csv('pima_indians_diabetes.csv')

#print(dataset.head(3))

"""plt.hist (dataset['bmi'].values, bins=100)
plt.xlabel('bmi')
plt.ylabel('number of people')
plt.show()"""

#split data into X(input variables) and Y(output variable)
X = ['preg','glu', 'bp','st' ,'ins','bmi','dpf','age']
Y = 'out'

#split data into train and test sets
seed = 7
test_size= 0.33
X_train, X_test, y_train, y_test = train_test_split(dataset[X],dataset[Y],test_size=test_size,random_state=seed)

#fit model on training data
model = XGBClassifier()
model.fit(X_train,y_train)

#make prediction for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

#evalute predictions
accuracy = accuracy_score(y_test,predictions)
print ( "accuracy: %.2f%%" % (accuracy*100.0))

