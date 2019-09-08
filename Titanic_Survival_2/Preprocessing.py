#installing dependencies
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

#Loading data
dataset = pd.read_csv('test.csv')
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

print (dataset.isnull())