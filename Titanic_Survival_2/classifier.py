#installing dependencies
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

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

"""
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)  
"""

model = RandomForestClassifier(n_jobs=2,random_state=0,n_estimators=30)
model.fit(dataset[X],dataset[Y])
#print (model)

y_pred = model.predict(X_test)

from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print(roc_auc)
#predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test,y_pred)
print ("accuracy: %.2f%%" % (accuracy*100.0))

#loading test data
test_dataset = pd.read_csv('test.csv')

test_dataset['Cabin'].fillna('Cs0',inplace=True)
test_dataset['Age'].fillna('-1',inplace=True)
test_dataset['Embarked'].fillna('A',inplace = True)
test_dataset['Fare'].fillna('0',inplace = True)

label_encoder_4 = LabelEncoder()
label_encoder_4 = label_encoder_4.fit(test_dataset['Sex'])
label_encoder_sex_1 = label_encoder_4.transform(test_dataset['Sex'])
test_dataset['Sex'] = label_encoder_sex_1

label_encoder_5 = LabelEncoder()
label_encoder_5 = label_encoder_5.fit(test_dataset['Cabin'])
label_encoder_cabin_1 = label_encoder_5.transform(test_dataset['Cabin'])
test_dataset['Cabin'] = label_encoder_cabin_1

label_encoder_6 = LabelEncoder()
label_encoder_6 = label_encoder_6.fit(test_dataset['Embarked'])
label_encoder_embarked_1 = label_encoder_6.transform(test_dataset['Embarked'])
test_dataset['Embarked'] = label_encoder_embarked_1

test_X = test_dataset[X]
#Predicting using above defined model
sur = model.predict(test_X)

#Writing into submission file
my_submission = pd.DataFrame({'PassengerId': test_dataset.PassengerId, 'Survived' : sur})
my_submission.to_csv('submission.csv', index =False)