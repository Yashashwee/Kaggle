#installing dependencies
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

#Loading data
dataset = pd.read_csv('train.csv')
dataset['Cabin'].fillna('Cs0',inplace=True)

#Preprocessing
label_encoder_1 = LabelEncoder()
label_encoder_1 = label_encoder_1.fit(dataset['Cabin'])
label_encoder_cabin = label_encoder_1.transform(dataset['Cabin'])
dataset['Cabin'] = label_encoder_cabin

label_encoder_2 = LabelEncoder()
label_encoder_2 = label_encoder_2.fit(dataset['Sex'])
label_encoder_sex = label_encoder_2.transform(dataset['Sex'])
dataset['Sex'] = label_encoder_sex

#splitting test data into input and output data
X=['Pclass','Sex','Age','SibSp','Parch']
Y = 'Survived'

#split data into train and test sets
test_size = 0.33
X_train, X_test, y_train , y_test = train_test_split ( dataset[X],dataset[Y],test_size=test_size,random_state=7)

#fit model on trainig data
model = XGBClassifier(base_score=0.5,
booster='gbtree', 
colsample_bylevel=1,
colsample_bytree=1,
gamma=0, 
learning_rate=0.1, 
max_delta_step=0,
max_depth=6, 
min_child_weight=1, 
missing=None, 
n_estimators=100,
n_jobs=1, 
nthread=None, 
objective='binary:logistic', 
random_state=0,
reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
silent=True, 
subsample=1)
model.fit(X_train,y_train)

"""
#Assess increment in performance with increment in trees
eval_set = [(X_train, y_train), (X_test, y_test)]
eval_metric = ["auc","error"]
model.fit(X_train, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=True)
"""

#make predictions based test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

#evaluate predictions
accuracy = accuracy_score(y_test,predictions)
print ("accuracy: %.2f%%" % (accuracy*100.0))

#loading test data
test_dataset = pd.read_csv('test.csv')

label_encoder_3 = LabelEncoder()
label_encoder_3 = label_encoder_3.fit(test_dataset['Sex'])
label_encoder_sex_1 = label_encoder_3.transform(test_dataset['Sex'])
test_dataset['Sex'] = label_encoder_sex_1

test_X = test_dataset[X]
#Predicting using above defined model
sur = model.predict(test_X)

#Writing into submission file
my_submission = pd.DataFrame({'PassengerId': test_dataset.PassengerId, 'Survived' : sur})
my_submission.to_csv('submission.csv', index =False)
