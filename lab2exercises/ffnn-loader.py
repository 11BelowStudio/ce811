
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np

iris_model: keras.Model = keras.models.load_model("iris_model")
"""
The iris model we made earlier.
"""

inputs_val = pd.read_csv('datasets/iris_test.csv',usecols = [0,1,2,3],skiprows = None,header=None).values
labels_val = pd.read_csv('datasets/iris_test.csv',usecols = [4],skiprows = None ,header=None).values.reshape(-1)

print("Data loaded (shapes only)", inputs_val.shape, labels_val.shape)
# Data loaded (shapes only) (30, 4) (30,)

model_outputs = iris_model(inputs_val)

np_outputs = model_outputs.numpy()

print("\nnumpy'd all outputs from model")
print(np_outputs)

simple_outputs = []

for o in np_outputs:
    highest_index = np.where(o == max(o))
    simple_outputs.append(highest_index[0][0])


print("\nSimplified version of all the outputs, just as the number for what flower it probably is")
print(simple_outputs)
print("Expected output:")
print(labels_val)