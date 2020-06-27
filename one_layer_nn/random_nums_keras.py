import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.datasets import load_digits
import sys

sys.path.append(".")
from utils.data_utils import *

np.random.seed(0)

set = np.random.randn(100, 8)
labels = compute_labels(set, 1)

train_set, train_labels, test_set, test_labels = split_train_test_dataset(set, labels, split_by="rows", perc=0.1)

model = Sequential()
model.add(Dense(4, input_dim=train_set.shape[1], activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_set, train_labels, epochs=100, batch_size=10, verbose=False)

scores = model.evaluate(test_set, test_labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

print(test_labels)
print(model.predict_classes(test_set).reshape(test_labels.shape))
