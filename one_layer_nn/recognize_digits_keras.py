import pandas as pd
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop, Adadelta, Adam
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib .pyplot as plt

def print_img(digits, indx):
    plt.gray()
    plt.matshow(dig.images[indx])
    plt.show()

dig = load_digits()

# print_img(dig, 922)
# 130 = 0
# 1174 = 7
# 265 = 9

onehot_target = pd.get_dummies(dig.target)
x_train, x_val, y_train, y_val = train_test_split(dig.data, onehot_target, test_size=0.1, random_state=20)

model = Sequential()
model.add(Dense(128, input_dim=x_train.shape[1], activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer=Adadelta(), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(x_train, y_train, epochs=50, batch_size=64, verbose=False)

scores = model.evaluate(x_train, y_train)
print("training set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

scores = model.evaluate(x_val, y_val)
print("test set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# preds = model.predict_classes(x_val)

# indx=0
# for index, row in y_val.iterrows():
#     if(row.tolist()[preds[indx]] == 0):
#         print("expected {} got {} at indx {} and index {}".format(row.tolist(), preds[indx], indx, index))
#         print_img(dig, index)
#     indx = indx + 1
