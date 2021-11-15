
import pickle
from typing import List, Tuple, TypeVar, Dict

from bots.rl18730 import PlayerRecord, PlayerRecordHolder, PlayerRecordNNEstimator, RoleAllocationEnum, GamestateTree, TPlayer

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import models


# https://datascience.stackexchange.com/questions/48796/how-to-feed-lstm-with-different-input-array-sizes

# https://keras.io/guides/understanding_masking_and_padding/

# TODO: load the PlayerRecordHolder pickle, use that data to train a model that can be put in PlayerRecordNNEstimator,
#   in order to estimate the likelihoods of each player being a spy.
#   validate the outputs of the model using the known spy info in the player records.
#   Remember to use the training set, test set, and validation set partitions present in PlayerRecordHolder,
#   to ensure that there's a good range of data being used.
# TODO: after that's done, save the model. Then, edit rl18730.py to load that model, and then use that model to
#   produce more refined player suspicion estimates.
# TODO: also compare the accuracy of the final neural network to the basic heuristic estimator already in PlayerRecord.
#   If the neural network turns out to be worse than the heuristic estimator, we don't bother with it.


