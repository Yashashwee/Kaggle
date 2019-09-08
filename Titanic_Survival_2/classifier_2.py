#installing dependencies
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
#Loading data
dataset = pd.read_csv('train.csv')
dataset['Cabin'].fillna('Cs0',inplace=True)
dataset['Age'].fillna('-1',inplace=True)
dataset['Embarked'].fillna('A',inplace = True)

#Preprocessing
label_encoder_1 = LabelEncoder()
label_encoder_1 = label_encoder_1.fit(dataset['Cabin'])
label_encoder_cabin = label_encoder_1.transform(dataset['Cabin'])
dataset['Cabin'] = label_encoder_cabin

label_encoder_2 = LabelEncoder()
label_encoder_2 = label_encoder_2.fit(dataset['Sex'])
label_encoder_sex = label_encoder_2.transform(dataset['Sex'])
dataset['Sex'] = label_encoder_sex

label_encoder_3 = LabelEncoder()
label_encoder_3 = label_encoder_3.fit(dataset['Embarked'])
label_encoder_embarked = label_encoder_3.transform(dataset['Embarked'])
dataset['Embarked'] = label_encoder_embarked

#splitting test data into input and output data
X=['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']
Y = 'Survived'

#split data into train and test sets
test_size = 0.33
X_train, X_test, y_train , y_test = train_test_split ( dataset[X],dataset[Y],test_size=test_size,random_state=7)

min_samples_leafs = np.linspace(0.1,0.5,5 ,endpoint=True)
train_results = []
test_results = []
for min_samples_leaf in min_samples_leafs:
   rf = RandomForestClassifier(min_samples_leaf = min_samples_leaf, n_jobs=-1)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = rf.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line_1 = plt.plot(min_samples_leafs, train_results, 'b', label='Train_AUC')
line_2 = plt.plot(min_samples_leafs, test_results, 'r', label='Test_AUC')

#plt.legend_handler(map={line_1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC_score')
plt.xlabel('min_samples_leaf')
plt.show()