from tensorflow import keras
import numpy as np
X = np.array([[3, 5, 2, 7], [4, 1, 3, 8], [6, 3, 8, 2], [9, 6, 1, 5]])
X = X.reshape(1, 4, 4, 1)
model_Conv2D = keras.models.Sequential()
model_Conv2D.add(keras.layers.Conv2D(1, (3, 3), strides=(1, 1), padding='valid', input_shape=(4, 4, 1)))
weights = [np.asarray([[[[1]], [[2]], [[1]]], [[[2]], [[1]], [[2]]], [[[1]], [[1]], [[2]]]]), np.asarray([0])]
model_Conv2D.set_weights(weights)
yhat = model_Conv2D.predict(X)
yhat.reshape(2, 2)
print(yhat)

X = yhat
model_Conv2D = keras.models.Sequential()
model_Conv2D.add(keras.layers.Conv2DTranspose(1, (3, 3), strides=(1, 1), padding='valid', input_shape=(2, 2, 1)))
weights = [np.asarray([[[[1]], [[2]], [[1]]], [[[2]], [[1]], [[2]]], [[[1]], [[1]], [[2]]]]), np.asarray([0])]
model_Conv2D.set_weights(weights)
yhat = model_Conv2D.predict(X)
yhat.reshape(4, 4)
print(yhat)
