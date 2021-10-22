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
# build a decision tree here.
# evaluated its accuracy on the validation set (should score>90%)
# plotted the decision tree and save it to a file called "decision-tree-iris.png"

clf: tree.DecisionTreeClassifier = tree.DecisionTreeClassifier(max_depth=2, min_samples_leaf=1)
clf.fit(inputs_train, labels_train)
output_predictions = clf.predict(inputs_val)

print("Output predictions : ", output_predictions)
print("Expected outputs   : ", labels_val)

print("Accuracy", accuracy_score(labels_val, output_predictions))

import matplotlib.pyplot as plt

fig = plt.figure()
tree.plot_tree(clf,
               feature_names=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)'],
               class_names=["setosa", "versicolor","virginica"],
               filled=True
               )
fig.savefig(
    "decision_tree_iris.png",
    bbox_inches="tight"
)

import pickle

pickle_out = open("iris_decision_tree.p","wb") # wb means write file, in binary format
pickle.dump(clf, pickle_out)
pickle_out.close()
