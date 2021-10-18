import pandas as pd

from numpy import where
from tensorflow import keras
from tensorflow.keras import layers

# Inputs: 1(?)
# Outputs: 1 (value of 0, 1, or 2)
    # 0: setosa
    # 1: versicolor
    # 2: virginica

#Load CSV files for iris datasets
inputs_train=pd.read_csv('datasets/iris_train.csv',usecols = [0,1,2,3],skiprows = None,header=None).values

labels_train = pd.read_csv('datasets/iris_train.csv',usecols = [4],skiprows = None ,header=None).values.reshape(-1)

inputs_val=pd.read_csv('datasets/iris_test.csv',usecols = [0,1,2,3],skiprows = None,header=None).values

labels_val = pd.read_csv('datasets/iris_test.csv',usecols = [4],skiprows = None ,header=None).values.reshape(-1)

print("Data loaded (shapes only)", inputs_train.shape, labels_train.shape, inputs_val.shape, labels_val.shape)
# Data loaded (shapes only) (120, 4) (120,) (30, 4) (30,)



import tensorflow as tf


num_inputs = 4 #TODO how many inputs does your neural network have?
    # inputs are in form of a 4-index 1D array.
num_outputs: int = 3 # It needs 3 outputs because there are 3 types of flowers being categorised.

#TODO  define a suitable keras model here

# Define Sequential model with 3 layers, architecture "numinputs-10-10-3", with tanh on all non final layers
model: keras.Model = keras.Sequential(name="flower_nn")

layer1: layers.Dense = layers.Dense(10, activation="tanh", input_shape=(4,))
model.add(layer1)

layer2: layers.Dense = layers.Dense(10, activation="tanh", input_shape=(10,))
model.add(layer2)

layer3: layers.Dense = layers.Dense(3, activation="tanh", input_shape=(10,))
model.add(layer3)

layer4: layers.Softmax = layers.Softmax()
model.add(layer4)


eta = 0.01
adam_opt = keras.optimizers.Adam(eta)
sgd_opt = keras.optimizers.SGD(eta)

using_adam: bool = True

epochs = 120

modelInfoString = "{} epochs, {} optimizer, {} eta".format(
    epochs, "adam" if using_adam else "sgd", eta
)

model.compile(
    #optimizer=keras.optimizers.Adam(0.01),
    #optimizer=keras.optimizers.SGD(0.001),
    optimizer=adam_opt if using_adam else sgd_opt,
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # remember this a classification problem!
    metrics='accuracy'
    # This allows the training process to keep track of how many flowers are being classified correctly.
)


history = model.fit(
    inputs_train,
    labels_train,
    batch_size=inputs_train.shape[0],
    epochs=epochs,
    validation_data=(inputs_val, labels_val),
)

import matplotlib.pyplot as plt

plt.title(modelInfoString)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"],label="Validation Set Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()


#model.save("iris_model")

model_outputs = model(inputs_val)

np_outputs = model_outputs.numpy()

print("\nnumpy'd all outputs from model")
print(np_outputs)

simple_outputs = []

for o in np_outputs:
    highest_index = where(o == max(o))
    simple_outputs.append(highest_index[0][0])


print("\nSimplified version of all the outputs, just as the number for what flower it probably is")
print(simple_outputs)
print("Expected output:")
print(labels_val)