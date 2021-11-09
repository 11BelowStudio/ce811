
import tensorflow as tf
import pandas as pd

from typing import List, Dict, Tuple, Union

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

df: pd.DataFrame = pd.read_json("../../logs/rl18730.log", lines=True, dtype='dict')

print(pd.DataFrame(df.teams))

teams_df: pd.DataFrame = pd.DataFrame(df.teams.to_dict())

#df = df.drop("teams", axis=1).join(teams_df)

print(df.head())

print(df.columns)

print(teams_df.head())

print(teams_df.count())

print(teams_df.columns)

print(teams_df[1])

# TODO:
#   * for each gamestate: [leader probs, spy probs] -> sabotage chance
#   * save them for each gamestate


def individual_gamestate_nn_processor(gs_ind: int, df: pd.DataFrame):

    filename: str = "rl18730_{}_nn".format(gs_ind)

    print(filename)

    all_teams = pd.DataFrame(df.teams.to_dict())

    print(all_teams.count())
    print(all_teams.columns)

    # flipping the rows and columns around
    all_teams_transposed = all_teams.T
    print("all transposed")
    print(all_teams_transposed.count())
    print(all_teams_transposed.columns)

    # All the team info things where there is data for a team at the index of gs_ind.
    selected_teams_transposed = all_teams_transposed[~all_teams_transposed["{}".format(gs_ind)].isna()]["{}".format(gs_ind)]

    print("selected transposed")
    print(selected_teams_transposed.count())
    print(selected_teams_transposed.head())

    """
    ok so basically selected_teams_transposed gives us
    
    row in main dataframe -> dict for the team that was used for the gamestate at the given index.    
    """

    # TODO:
    #   filter out sabotage -1(?)
    #   make a NN to work out if the given spy probabilities lead to a sabotaged mission or a successful mission
    #       actually, wait, might be better to do it like spy probs -> win/loss overall(?)
    #           or just have a second NN along with it that goes gamestate outcome -> win/loss overall(?)
    #   save that NN.

    #print(selected_teams_transposed.columns)

    #selected_teams = selected_teams_transposed.T

    #print("selected untransposed")
    #print(selected_teams.count())
    #print(selected_teams.head())
    #print(selected_teams.columns)

    # all_teams[n] gets the teams for the nth iteration



individual_gamestate_nn_processor(-7, df)


