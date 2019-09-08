import pandas as pd
from sklearn.preprocessing import LabelEncoder

#load data
dataset = pd.read_csv('train.csv')
dataset['Cabin'].fillna('Cs0',inplace=True)

label_encoder_1 = LabelEncoder()
label_encoder_1 = label_encoder_1.fit(dataset['Cabin'])
label_encoder_cabin = label_encoder_1.transform(dataset['Cabin'])
dataset['Cabin'] = label_encoder_cabin

label_encoder_2 = LabelEncoder()
label_encoder_2 = label_encoder_2.fit(dataset['Sex'])
label_encoder_sex = label_encoder_2.transform(dataset['Sex'])
dataset['Sex'] = label_encoder_sex
