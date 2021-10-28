import tensorflow.python.framework.ops

from player import Bot, TPlayer
from game import State
import random

from typing import List, Dict, Iterable, Tuple, Any, Set

# to run this
# python3 competition.py 1000 bots/beginners.py bots/neuralbot.py

import tensorflow as tf
from tensorflow import keras
model: keras.Model = keras.models.load_model('bots/lab3/loggerbot_classifier')
import numpy as np
import sys

from loggerbot import LoggerBot # this assumes our loggerbot was in a file called loggerbot.py

class NeuralBot(LoggerBot):

    def calc_player_probabilities_of_being_spy(self) -> Dict[TPlayer, np.float32]:
        # This loop could be made much more efficient if we push all player's input patterns
        # through the neural network at once, instead of pushing them through one-by-one
        probabilities: Dict[TPlayer, np.float32] = {}
        for p in self.game.players:
            # This list comprising the input vector must build in **exactly** the same way as
            # we built data to train our neural network - otherwise the neural network
            # is not being used to approximate the same function it's been trained to model.
            # That's why this class inherits from the class LoggerBot-
            #   so we can ensure that logic is replicated exactly.
            input_vector =\
                [self.game.turn, self.game.tries, p.index, p.name, self.missions_been_on[p], self.failed_missions_been_on[p]]+self.num_missions_voted_up_with_total_suspect_count[p]+self.num_missions_voted_down_with_total_suspect_count[p]
            input_vector: List[int] = input_vector[4:]
            # remove the first 4 cosmetic details, as we did when training the neural network

            input_vector: np.ndarray = np.array(input_vector).reshape(1,-1)
            # change it to a rank-2 numpy array, ready for input to the neural network.

            output = model(input_vector) # run the neural network
            output_probabilities = tf.nn.softmax(output, axis=1)
            # The neural network didn't have a softmax on the final layer,
            # so I'll add the softmax step here manually.

            probabilities[p] = output_probabilities[0, 1].numpy()
            # this [0,1] pulls off the first row (since there is only one row)
            # and the second column
            # (which corresponds to probability of being a spy;
            # the first column is the probability of being not-spy)

        return probabilities
        # This returns a dictionary of {player: spyProbability}

    def select(self, players: List[TPlayer], count: int) -> List[TPlayer]:
        # here I'm replicating logic we used in the CountingBot exercise of lab1-challenge3.
        # But instead of using the count as an estimation of how spy-like a player is, instead
        # we'll use the neural network's estimation of the probability.
        spy_probs: Dict[TPlayer, np.float32] = self.calc_player_probabilities_of_being_spy()
        sorted_players_by_trustworthiness: List[TPlayer] = \
            [k for k, v in sorted(spy_probs.items(), key=lambda item: item[1])]
        if self in sorted_players_by_trustworthiness[:count]:
            result: List[TPlayer] = sorted_players_by_trustworthiness[:count]
        else:
            result: List[TPlayer] = [self] + sorted_players_by_trustworthiness[:count-1]
        return result

    def vote(self, team: List[TPlayer]) -> bool:
        spy_probs: Dict[TPlayer, np.float32] = self.calc_player_probabilities_of_being_spy()
        sorted_players_by_trustworthiness: List[TPlayer] = \
            [k for k, v in sorted(spy_probs.items(), key=lambda item: item[1])]
        if not self.spy:
            for x in team:
                if x in sorted_players_by_trustworthiness[-2:]:
                    return False
            return True
        else:
            return True

    def sabotage(self) -> bool:

        # note to self: might try to work on a better neural network thing for the assignment

        # don't bother on turn 1
        if self.game.turn == 1:
            return False
        # if the resistance have won twice
        if self.game.wins == 2:
            # see if it's my job to sabotage
            return self.isItMyJobToSabotage()
        # in all other cases
        else:
            # sorts players by how sus they are, in ascending order
            spy_chances: Dict[TPlayer, float] = self.announce()
            sorted_by_sus_level: List[TPlayer] = \
                sorted(self.game.players, key=lambda p1: spy_chances[p1])

            # sorts players by their failed missions, in ascending order.
            # sorted_by_failed_missions: List[TPlayer] = \
            #    sorted(self.game.players, key=lambda p1: self.failed_missions_been_on[p1])
            most_legit_three: Set[TPlayer] = set(sorted_by_sus_level[:3])
            # set(sorted_by_failed_missions[:3])

            # look at all the other people in the  team
            others_in_team: Set[TPlayer] = self.game.team.copy()
            others_in_team.remove(self)
            # if the team members are all legit overall (all in the top 3 of legitness)
            if len(set(others_in_team) & set(most_legit_three)) == len(others_in_team):
                # see if this bot is responsible for sabotaging this.
                return self.isItMyJobToSabotage()
        return True

        #sorted_by_failed_missions: List[TPlayer] = \
        #    [p for p, v in sorted(self.failed_missions_been_on.keys(), key=lambda k: self.failed_missions_been_on[k])]
        # ascending order
        #if len(set(self.game.team) - set(sorted_by_failed_missions[:3])) == 0:
        #    # if the team is overall relatively legit (team contains the fewest 3 failed)
        #    return True
        #
        #return False

    def isItMyJobToSabotage(self) -> bool:
        """
        Works out whether or not this bot is responsible for sabotaging
        this round.
        :return: true if this bot should sabotage, false if it's someone else's problem.
        """
        # if the leader's a spy, it's their job to sabotage.
        if self.game.leader == self:
            # if this bot is the leader, it'll sabotage
            return True
        if self.game.leader in self.spies:
            # if the leader is another spy, it's their job to sabotage
            return False
        # if the leader isn't a spy
        else:
            # it's the job of the first spy in the team to sabotage
            for p in self.game.team:
                if p == self:
                    # if this bot is the first spy in the team, it sabotages
                    return True
                elif p in self.spies:
                    # otherwise, it's that spy's job
                    break
            return False

    def announce(self) -> Dict[TPlayer, float]:
        """Publicly state beliefs about the game's state by announcing spy
        probabilities for any combination of players in the game.  This is
        done after each mission completes, and takes the form of a mapping from
        player to float.  Not all players must be specified, and of course this
        can be innacurate!

        :return: Dict[TPlayer, float]     Mapping of player to spy probability.
        """
        spy_or_not_spy: Dict[TPlayer, np.float32] = \
            self.calc_player_probabilities_of_being_spy()
        spy_chances: Dict[TPlayer, float] = {}
        for p in spy_or_not_spy.keys():
            spy_chances[p] = float(spy_or_not_spy[p])
        return spy_chances

    ''' The 3 methods onVoteComplete, onGameRevealed, onMissionComplete
    will inherit their functionality from ancestor.  We want them to do exactly 
    the same as they did when we captured the training data, so that the variables 
    for input to the NN are set correctly.  Hence we don't override these methods
    '''
    
    # This function used to output log data to the log file. 
    # We don't need to log any data any more so let's override that function
    # and make it do nothing...
    def onGameComplete(self, win: bool, spies: List[TPlayer]) -> None:
        pass


