import numpy as np
import pandas as pd

# Getting Data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Data Exploration / Analysis
train_df.info()
train_df.describe()

# Missing Data

# Cabin
data = [train_df, test_df]

def getDeck(x) :
    return x[0:1]

for dataset in data :
    dataset['Cabin'].fillna('U0', inplace=True)
    dataset['Deck'] = dataset.Cabin.apply(getDeck)

train_df = train_df.drop(columns = ['Cabin'])
test_df = test_df.drop(columns = ['Cabin'])

# Age
survived_mean = train_df[train_df.Survived == 1].Age.mean()
not_survived_mean = train_df[train_df.Survived == 0].Age.mean()
test_mean = test_df.Age.mean()

train_df.update(train_df[train_df.Survived==1].Age.fillna(survived_mean))
train_df.update(train_df[train_df.Survived==0].Age.fillna(not_survived_mean))
test_df.Age.fillna(test_mean, inplace=True)

# Embarked
data = [train_df, test_df]
for dataset in data :
    dataset.Embarked.fillna('S', inplace=True)
    
# Drop Ticket and Name
train_df = train_df.drop(columns=['Ticket'])
test_df = test_df.drop(columns=['Ticket'])
train_df = train_df.drop(columns=['Name'])
test_df = test_df.drop(columns=['Name'])

# Getting Survived Column
y_df = train_df.Survived
train_df = train_df.drop(columns = ['Survived'])

# Categorical Data
def sex(x) :
    if x == 'male' :
        return 1
    else :
        return 0

def embarked(x) :
    if x == 'S' :
        return 0
    elif x == 'C' :
        return 1
    else:
        return 2

def deck(x) :
    dic = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'U':7}
    return dic[x]

def hello(x) :
    if x == 'T' :
        return 'A'
    else :
        return x

train_df.Deck = train_df.Deck.apply(hello)

for i in range(len(train_df.values)) :
    if train_df.iloc[i, 7] == 'T' :
        print(i)

data = [train_df, test_df]
for dataset in data :
    dataset['Sex'] = dataset['Sex'].apply(sex)
    dataset['Embarked'] = dataset['Embarked'].apply(embarked)
    dataset['Deck'] = dataset['Deck'].apply(deck)
    
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[1, 6, 7])
onehotencoder.fit(train_df)

train_df = onehotencoder.transform(train_df).toarray()
test_df = onehotencoder.transform(test_df).toarray()
    
from sklearn.linear_model import LogisticRegression
algo = LogisticRegression()
algo.fit(train_df, y_df)
y_pred = algo.predict(test_df)


np.savetxt('sol.csv', y_pred, fmt="%d")