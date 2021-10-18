import pandas as pd

#Load CSV files for iris datasets:

# Inputs: 1(?)
# Outputs: 1 (value of 0, 1, or 2)
    # 0: setosa
    # 1: versicolor
    # 2: virginica

inputs_train = pd.read_csv('datasets/iris_train.csv',usecols = [0,1,2,3],skiprows = None,header=None).values

labels_train = pd.read_csv('datasets/iris_train.csv',usecols = [4],skiprows = None ,header=None).values.reshape(-1)
inputs_val = pd.read_csv('datasets/iris_test.csv',usecols = [0,1,2,3],skiprows = None,header=None).values
labels_val = pd.read_csv('datasets/iris_test.csv',usecols = [4],skiprows = None ,header=None).values.reshape(-1)

print("Data loaded (shapes only)", inputs_train.shape, labels_train.shape, inputs_val.shape, labels_val.shape)

print(inputs_train[0])

# Data loaded (shapes only) (120, 4) (120,) (30, 4) (30,)

from sklearn import tree
from sklearn.metrics import accuracy_score
# TODO build a decision tree here.
# TODO evaluate its accuracy on the validation set (should score>90%)
# TODO plot your decision tree and save it to a file called "decision-tree-iris.png"
