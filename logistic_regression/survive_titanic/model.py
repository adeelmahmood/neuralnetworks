import pandas as pd
from pandas import Series, DataFrame

import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')

# train = train.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

# train["Embarked"] = train["Embarked"].fillna("S")

print(train.head(8))
print(train.info())
print(train.describe())
