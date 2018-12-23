import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt #graphing
from sklearn import preprocessing

np.random.seed(1)

model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')

vals=30
X = np.array([[i] for i in range(vals)])
y = np.array([j+5 for j in X]) + np.random.uniform(0, 0.5, (vals, 1))
normalized_X = preprocessing.scale(X)
model.fit(normalized_X, y, epochs=1500)
y_predicted = model.predict(normalized_X)
plt.scatter(normalized_X, y)
plt.plot(normalized_X, y_predicted)
plt.show()
