import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score

import pickle

# loading the validation set
inputs_val = pd.read_csv('datasets/iris_test.csv',usecols = [0,1,2,3],skiprows = None,header=None).values
labels_val = pd.read_csv('datasets/iris_test.csv',usecols = [4],skiprows = None ,header=None).values.reshape(-1)

print("Data loaded (shapes only)", inputs_val.shape, labels_val.shape)

# opening the pickle that the decision tree was saved in
pickle_in = open("iris_decision_tree.p","rb")
my_clf = pickle.load(pickle_in) # and actually loading it

# running that loaded decision tree
output_predictions_loaded=my_clf.predict(inputs_val)
print("Accuracy (from loaded tree)", accuracy_score(labels_val, output_predictions_loaded))
