import os
import pickle
import random

from player import Bot, Player
from game import State

from typing import TypeVar, List, Dict, Set, Tuple, Iterable, FrozenSet, Union

from enum import Enum

from pathlib import Path

import json

TPlayer = TypeVar("TPlayer", bound="Player")
"""A generic type for anything that might be a Player object"""

T = TypeVar("T")
"""A generic type that could be anything."""

resources_file_path: Path = Path(__file__).parent/"rl18730"
"""
A file path for resources for this bot
USAGE:
    with open(resources_file_path/"filename",etc) as f:
        file stuff goes here
        f.close()
        

"""


# Hi, my name is Rachel, and welcome to Jackass.
# at least there are type annotations :^)


"""
Scrap that current stuff, taking another look at the DeepRole paper.

public game tree is a history of third-person observations, o ∈ O(h), instead of just actions.
    public actions
        * nominations?
        * sabotages?
        * votes?
    observable consequences of actions
        * how did the vote affect the outcome?
        * overall win/loss?
        
we maintain a human-interpretable joint posterior belief b(ρ|h) over the initial assignment of roles ρ.
ρ represents a full assignment of roles to players (the result of the initial
chance action) – so our belief b(ρ|h) represents the joint probability that each player has the role
specified in ρ, given the observed actions in the public game tree.

    h given p
    
    p(p|h) = p(p1 ^ p2 ^ p3 ^ p4 ^ p5|h)
    
    p(h|p) = 1
    
    p(p|h) = p(h|p) * p(p)/p(h)
    
    
note: would need to zero the probability of any outcome logically inconsistent with the game tree.



neural network:
    (proposer, public belief values for this gamestate)
    -> 2 dense layers (15(?))
    -> win layer (10)
        win(role(p))
            Probability of a resistance win for each role assignment at this gamestate(?)
    -> probability-weighted values for each info set
        5 * 2 ([isRes, isSpy] for each player)



While it’s possible to estimate these values using a generic feed-
forward architecture, it may cause lower sample efficiency, require longer training time, or fail
to achieve a low loss. We design an interpretable custom neural network architecture that takes
advantage of restrictions imposed by the structure of many hidden role games. Our network feeds
a one-hot encoded vector of the proposer player i and the belief vector b into two fully-connected
hidden layers of 80 ReLU units. These feed into a fully-connected win probability layer with sigmoid
activation. This layer is designed to take into account the specific structure of V , respecting the binary
nature of payoffs in Avalon (players can only win or lose). It explicitly represents the probability of a
Resistance win (~w = P(win|ρ)) for each assignment ρ








"""



class RoleAllocationEnum(Enum):
    """An enumeration for each possible permutation of role allocations.
    Values are worked out via boolean values (true = spy, false = not spy)

    For example, AC = [True, False, True, False, False] = 1 + 4 -> refers to items at indexes 0 and 3."""
    AB: Tuple[bool, bool, bool, bool, bool] = (True, True, False, False, False)
    AC: Tuple[bool, bool, bool, bool, bool] = (True, False, True, False, False)
    AD: Tuple[bool, bool, bool, bool, bool] = (True, False, False, True, False)
    AE: Tuple[bool, bool, bool, bool, bool] = (True, False, False, False, True)
    BC: Tuple[bool, bool, bool, bool, bool] = (False, True, True, False, False)
    BD: Tuple[bool, bool, bool, bool, bool] = (False, True, False, True, False)
    BE: Tuple[bool, bool, bool, bool, bool] = (False, True, False, False, True)
    CD: Tuple[bool, bool, bool, bool, bool] = (False, False, True, True, False)
    CE: Tuple[bool, bool, bool, bool, bool] = (False, False, True, False, True)
    DE: Tuple[bool, bool, bool, bool, bool] = (False, False, False, True, True)

    def extract_sublist_from(self, input_list: List[T]) -> List[T]:
        """
        Use the RoleAllocationEnum to extract the appropriate sublist from the provided list of objects
        :param input_list: the list of all 5 input objects, referring to players,
        indexed according to the game.state.players ordering.
        :return: A list consisting of the two objects from the input list that this RoleAllocationEnum is referring to.
        """
        return [input_list[p] for p in range(0, len(input_list)) if self.value[p]]

    def get_value(self) -> Tuple[bool, bool, bool, bool, bool]:
        return self.value

    def to_string_for_json(self) -> str:
        """
        Turns the value of this into a string for usage in json.dumps() stuff
        :return: a string ab, where a is index of first true, and b is index of second true
        """
        return "".join(["1" if i else "0" for i in self.value])

    def from_jsoned_string(self, jsoned_string: str) -> "RoleAllocationEnum":
        """
        from a string ab (a = index of first true, b is index of second true),
        as would have been returned by to_string_for_json, and then returns the
        RoleAllocationEnum with the value that the string ab describes.
        Throws an exception if an unexpected input is given.
        :param jsoned_string: the sort of string that to_string_for_json would expect to be given
        :return: the RoleAllocationEnum which that string describes.
        """
        return RoleAllocationEnum(tuple([True if n == "1" else False for n in jsoned_string]))


class GamestateTree(object):
    """
    In short, this is a data structure that basically exists to store info about gamestate history,
    with a tree of possible indexed gamestates (with info about which indexes that gamestate could lead to).

    Everything in here is all static-read only stuff, used for reference by everything else.

    win offset + 3
    loss: offset - 1

                -3
            -2      -3 (0)
        -1      1       -3 (3)
    0       2       4
        3       5       7
            6       7 (8)
                7 (9)



    here's a crappy visual representation of how I've indexed the gamestates.
    up: losing. down: winning.


    after a loss, next round's attempt indexes follow on from 'currentRoundFinalAttempt + 1'
    after a win, next round's attempt indexes start from 'currentRoundFinalAttempt + 16'
    -15 (lowest possible value) is 'spy victory'.
    35 (highest possible value) is 'resistance victory'.

    Anyway, I've done this for use as a heuristic gamestate value estimate, for use when there isn't
    enough data for use by a monte carlo tree search algorithm or anything like that.

    Resistance wants to get to the highest possible index.
    Spies want to get to the smallest possible index.
    Indexes for individual rounds are in descending order (1st nomination is highest, 5th nomination is lowest)

    easiest way of assigning keys to them in a way that might make a little bit of sense.
    m1  m2  m3  m4  m5
                -15 (spy win)
            -10,-14 -15 (spy win)
        -5,-9   5,1     -15 (spy win)
    0,-4    10,6    20,16
        15,11   25,21   35 (resistance win)
            30,26   35 (resistance win)
                35 (resistance win)
    """

    # yep. we got an an inner class. brace yourself, it gets worse.
    class GamestateTreeNode(object):
        """
        Contains indexes of what gamestate nodes this gamestate can lead to

        Recalls what index within the tree this index has,
        along with int pointers to the indexes that hold the nodes
        which must be navigated to by the tree traversal algorithm
        when either this proposal is rejected, or when the mission associated with this
        proposal passes or fails.

        Might try to use this to help structure some neural networks or monte carlo search trees
        to indicate what is likely to happen at this point (proposal fail/mission pass/mission fail)
        given the history up to this point, and what the best way to vote for the mission would be.
        """

        def __init__(self, index: int, voteFailedChild: int, missionPassedChild: int, missionFailedChild: int):
            self._index = index
            self._voteFailedChild: int = voteFailedChild
            self._missionPassedChild: int = missionPassedChild
            self._missionFailedChild: int = missionFailedChild
            self._team_size: int = 3
            indMinus1Mod5: int = (index - 1) % 5
            if indMinus1Mod5 > 4 and index - indMinus1Mod5 != 11:
                # basically everything that's range(0, -5, -1) has a team size of 2, except range(20, 16, -1)
                self._team_size = 2

        def __str__(self):
            return "Index: {:3d}, Reject {:3d}, Pass {:3d}, Fail {:3d}, Team {:2d}" \
                .format(self._index, self._voteFailedChild, self._missionPassedChild,
                        self._missionFailedChild, self._team_size)

        @property
        def index(self) -> int:
            """index of this node"""
            return self._index

        @property
        def voteFailedChild(self) -> int:
            """index of state reached if this vote fails"""
            return self._voteFailedChild

        @property
        def missionPassedChild(self) -> int:
            """index of state reached if this mission passes"""
            return self._missionPassedChild

        @property
        def missionFailedChild(self) -> int:
            """index of state reached if this mission fails"""
            return self._missionFailedChild

        @property
        def hammer(self) -> bool:
            """
            Whether or not this node is hammer.
            (final nomination attempt).
            """
            return self._missionFailedChild == self._voteFailedChild

        @property
        def team_size(self) -> int:
            """The size of the team for this mission"""
            return self._team_size

    _spy_win_offset: int = -5
    """Offset for spy wins"""
    _res_win_offset: int = 15
    """Offset for resistance wins"""

    _spy_win_index: int = -15
    """Spy win state is index -15"""
    _res_win_index: int = 35
    """Resistance win state is index 35"""

    _node_dict: Dict[int, "GamestateTree.GamestateTreeNode"] = {}
    """The dictionary that holds the indexed GamestateTreeNode objects"""

    # and now it's time for some static initialization of these class attributes.

    for i in range(-2, 7):
        # 9 'groups' of actual gamestates that matter (-2 to 6 are the round IDs, with nomination nodes being
        # in the range -14,30

        nom1: int = i * 5
        rangeEnd: int = nom1 - 5
        step: int = -1

        for g in range(nom1, rangeEnd, step):
            # creates a gamestate index node for the current proposal located at index g.

            # failing mission goes to the index that's after the failure one
            spyWinIndex = rangeEnd
            if spyWinIndex == 0 or spyWinIndex == 15:  # reaching these via a loss means that spies have won 3 times.
                spyWinIndex = _spy_win_index
            resWinIndex = nom1 + 15
            if resWinIndex > 30:  # any index over 30 means resistance won 3 times
                resWinIndex = _res_win_index

            failIndex = g + step  # failed nomination -> go to index of next step
            if failIndex == rangeEnd:  # if the next step is at the end of range, go to loss instead
                failIndex = spyWinIndex

            _node_dict[g] = GamestateTreeNode(g, failIndex, resWinIndex, spyWinIndex)

    @classmethod
    def encode_gamestate_index(cls, res_wins: int, spy_wins: int, nomination_attempt: int) -> int:
        """
        Work out the what gamestate index is associated with the current state of the game.
        Also it works as an int to measure the value of the gamestate. higher = better for resistance.
        Note: if one team have won 3 times, it returns their win node index.
        :param res_wins: how many times the resistance have won so far (0-2)
        :param spy_wins: how many times the spies have won so far (0-2)
        :param nomination_attempt: which attempt at nomination this is (1-5)
        :return: the appropriate index for the current gamestate.
        """
        if res_wins == 3:
            return cls._res_win_index
        elif spy_wins == 3:
            return cls._spy_win_index
        return (res_wins * cls._res_win_offset) + \
               (spy_wins * cls._spy_win_offset) - \
               (nomination_attempt - 1)  # nom attempts are numbered 1-5. nom1 has highest index, nom5 has lowest.

    @classmethod
    def get_index_from_gamestate_object(cls, state: State) -> int:
        """
        A wrapper for encode_gamestate_index that a game.state method can be passed to to obtain the current
        value of the gamestate
        :param state:
        :return:
        """
        return cls.encode_gamestate_index(state.wins, state.losses, state.tries)

    @classmethod
    def decode_gamestate_index(cls, index: int) -> Tuple[int, int, int]:
        """
        Given an index, returns the tuple with the current resistance wins, spy wins, and current nomination
        :param index: gamestate index to decode
        :return: tuple (resistance wins, spy wins, current nomination)
        """

        if index == cls._spy_win_index:  # if it's the spy win state, just return 3 spy wins
            return 0, 3, 0
        elif index == cls._res_win_index:  # if it's the resistance win state, just return 3 resistance wins
            return 3, 0, 0

        nom: int = index % 5  # start by getting index % 5
        round_id: int = index - nom  # round id will refer to index for nomination 1 for that round (for now)
        nom = 6 - nom  # 1st currently has nom 0, 5th has nom 4, so we fix that
        if nom == 6:
            nom = 1  # 6-0 is 6. 1st nom currently 0 -> turned to 6
        else:
            round_id += 5  # if this wasn't 1st nom, round id refers to the wrong 1st nom, so we fix that.

        round_id = round_id // 5
        # int division by 5, get an actual round id (refer to the above diagrams)

        return (round_id + 2) // 3, -round_id % 3, nom

    @classmethod
    def get_gstnode_from_index(cls, index: int) -> "GamestateTree.GamestateTreeNode":
        """
        Obtain the gamestate node at the given index. If you don't know what index you want,
        work it out via encode_gamestate_index
        :param index: index of node that you want to get.
        :return: that gamestate info node.
        """
        return cls._node_dict[index]

    # my face mfw static properties aren't a thing in 3.8 i crie evertiem
    @classmethod
    def get_res_win_index(cls) -> int:
        """Get the index associated with the resistance winning"""
        return cls._res_win_index

    @classmethod
    def get_spy_win_index(cls) -> int:
        """Get the index associated with the spies winning"""
        return cls._spy_win_index

    @classmethod
    def get_all_possible_gamestate_indices(cls) -> List[int]:
        """
        Get all the possible indices for gamestates
        """
        the_indexes: List[int] = [*cls._node_dict.keys()]
        the_indexes.extend([cls._res_win_index, cls._spy_win_index])
        return the_indexes

    @classmethod
    def get_all_non_terminal_gamestates_indices(cls) -> List[int]:
        """
        List of all the gamestate indices, not including the win state indices
        :return: list of gamestate indices
        """
        return [*cls._node_dict.keys()]

    @classmethod
    def get_hammer_indices(cls) -> List[int]:
        """
        Indices for the final nomination attempts
        """
        return [k for k in cls._node_dict.keys() if cls._node_dict[k].hammer]

    @classmethod
    def get_team_size_from_index(cls, ind: int) -> int:
        """Returns the team size for the gamestate at the given index.
        If given gamestate is not a known gamestate, this just returns 3 instead because I'm lazy."""
        # indMinus1Mod5: int = (ind-1) % 5
        # if indMinus1Mod5 > 4 and ind - indMinus1Mod5 != 11:
        #    # basically everything that's range(0, -5, -1) has a team size of 2, except range(20, 16, -1)
        #    return 2
        # return 3
        if ind in cls._node_dict.keys():
            return cls._node_dict[ind].team_size
        else:
            return 3

    @classmethod
    def get_raw_regret_from_index(cls, ind: int) -> Dict[str, int]:
        """
        Works out 'raw' counterfactual regret for the actions that could follow from this gamestate.
        I'm referring it it as 'raw', because it's just the regret associated with each outcome
        (success/sabotaged/rejection) from the current gamestate, and something else will be
        calculating the actual probabilities of each outcome happening (which these values can simply be multiplied
        by later on)
        :param ind: index of the parent node that we're trying to work out the immediate counterfactual regret of
        :return: dict with "reject", "pass", and "fail" regret values.
        """

        if ind == cls._spy_win_index or ind == cls._res_win_index:
            return {"reject": 0, "pass": 0, "fail": 0}  # not much left to regret if the game is already over.

        this_state_node: "GamestateTree.GamestateTreeNode" = cls._node_dict[ind]

        return {"reject": this_state_node.voteFailedChild - ind,
                "pass": this_state_node.missionPassedChild - ind,
                "fail": this_state_node.missionFailedChild - ind}


# a little bit of cleanup on the gamestatetree, removing a couple of unwanted static variables that hung around
# noinspection PyBroadException
try:
    delattr(GamestateTree, "nom1")
    delattr(GamestateTree, "rangeEnd")
    delattr(GamestateTree, "step")
except Exception:
    pass


class MCTSTree(object):
    """
    A monte carlo search tree, that uses the basic tree structure of the above gamestatetree stuff,
    and is intended to work out the likelihood of each outcome from each state in that tree
    """

    class MCTSNode(object):
        """A node in this MCTS tree"""

        def __init__(self, state_index: int):
            self._index: int = state_index
            self._res_wins_from_node: int = 0
            self._sims_from_node: int = 0

        @property
        def n_index(self) -> int:
            """The gamestatetree index of this node"""
            return self._index

        @property
        def n_outcomes(self) -> Dict[str, int]:
            """Returns the indices of the nodes that could be the outcomes for this node
            :return: dict with "reject", "pass", and "fail" node indices"""
            gs_node: GamestateTree.GamestateTreeNode = GamestateTree.get_gstnode_from_index(self._index)
            return {"reject": gs_node.voteFailedChild,
                    "pass": gs_node.missionPassedChild,
                    "fail": gs_node.missionFailedChild}

        @property
        def res_wins(self) -> int:
            """How many times the resistance won from this node"""
            return self._res_wins_from_node

        @property
        def spy_wins(self) -> int:
            """How many times the spies won from this node"""
            return self._sims_from_node - self._res_wins_from_node

        @property
        def sims(self) -> int:
            """How many times simulations have been run with this node"""
            return self._sims_from_node

        def ran_simulation(self, resistance_win: bool) -> None:
            """
            Call this to update the node with new simulation info after reaching the end of a simulation
            :param resistance_win: whether or not the resistance won
            :return: true if they won, false otherwise.
            """
            self._sims_from_node += 1
            if resistance_win:
                self._res_wins_from_node += 1



    def __init__(self):
        """Attempts to initialize the MCTS tree"""

        self._mcts_tree: Dict[int, "MCTSTree.MCTSNode"] = {}
        """Dictionary of MCTS nodes"""

        for i in GamestateTree.get_all_non_terminal_gamestates_indices():
            self._mcts_tree[i] = MCTSTree.MCTSNode(i)



class TeamRecord(object):
    """
    A record of the teams that have been nominated, who nominated them, whether or not the nomination passed,
    the outcome of the mission, and who voted in favour of the team.

    Properties are all read-only.
    """

    def __init__(self,
                 team: Iterable[TPlayer],
                 leader: TPlayer,
                 mission_number: int,
                 nomination_attempt: int,
                 nomination_successful: bool,
                 sabotages: int,
                 voted_for_team: Iterable[TPlayer],
                 prior_predicted_spy_probabilities: Dict[TPlayer, float],
                 latter_predicted_spy_probabilities: Dict[TPlayer, float]
                 ):
        """
        Constructor for this record of team information
        :param team: who was on the team?
        :param leader: who was the leader?
        :param mission_number: which mission was this for?
        :param nomination_attempt: which nomination attempt was this?
        :param nomination_successful: did the nomination pass?
        :param sabotages: how many times was the mission sabotaged?
            (will be overwritten by -1 if nomination unsuccessful)
        :param voted_for_team: who voted in favour of the team?
        :param prior_predicted_spy_probabilities: predicted probabilities of each player being a spy,
        as of the start of this turn
        :param latter_predicted_spy_probabilities: predicted probabilities of each player being a spy,
         as of the end of this turn.
        """

        self._team: Tuple[TPlayer, ...] = tuple(team)
        self._leader: TPlayer = leader
        self._mission_number: int = mission_number
        self._nomination_attempt: int = nomination_attempt
        self._nomination_successful: bool = nomination_successful
        self._sabotages: int = sabotages if nomination_successful else -1
        self._voted_for_team: FrozenSet[TPlayer] = frozenset(voted_for_team)
        self._prior_predicted_spy_probabilities: Dict[TPlayer, float] = prior_predicted_spy_probabilities.copy()
        self._latter_predicted_spy_probabilities: Dict[TPlayer, float] = latter_predicted_spy_probabilities.copy()
        #self._predicted_spy_probabilities: Dict[TPlayer, float] = {}
        #for p in [*predicted_spy_probabilities.keys()]:
        #    self._predicted_spy_probabilities[p] = predicted_spy_probabilities[p]


    @property
    def team(self) -> Tuple[TPlayer, ...]:
        """Who was on the team?"""
        return self._team

    @property
    def leader(self) -> TPlayer:
        """Who nominated the team?"""
        return self._leader

    @property
    def mission_number(self) -> int:
        """What mission was this for?"""
        return self._mission_number

    @property
    def nomination_attempt(self) -> int:
        """Which nomination was this?"""
        return self._nomination_attempt

    @property
    def nomination_successful(self) -> bool:
        """Did the nomination pass?"""
        return self._nomination_successful

    @property
    def sabotages(self) -> int:
        """How many times was the mission sabotaged? (-1 if it was rejected instead)"""
        return self._sabotages

    @property
    def voted_for_team(self) -> FrozenSet[TPlayer]:
        """Who voted in favour of the team?"""
        return self._voted_for_team

    @property
    def prior_predicted_spy_probabilities(self) -> Dict[TPlayer, float]:
        """What were the predicted probabilities of each player on the team being a spy?"""
        return self._prior_predicted_spy_probabilities.copy()

    @property
    def latter_predicted_spy_probabilities(self) -> Dict[TPlayer, float]:
        """What were the predicted probabilities of each player on the team being a spy?"""
        return self._latter_predicted_spy_probabilities.copy()

    @property
    def public_belief_states_prior(self) -> Dict[RoleAllocationEnum, float]:
        """Public belief states(?) for this gamestate (before the outcome of this mission attempt)."""

        player_kv: List[Tuple[TPlayer, float]] = [*self._prior_predicted_spy_probabilities.items()]

        public_state_dict: Dict[RoleAllocationEnum, float] = {}

        for rp in RoleAllocationEnum.__members__.values():
            chance_of_this_allocation: float = 1
            rp_list = rp.get_value()
            for r in range(0, len(rp_list)):
                if rp_list[r]:
                    chance_of_this_allocation *= player_kv[r][1]
                else:
                    chance_of_this_allocation *= (1 - player_kv[r][1])
            public_state_dict[rp] = chance_of_this_allocation

        total_chances: float = sum(public_state_dict.values())

        if total_chances == 0:
            total_chances = 1

        for k in public_state_dict.keys():
            public_state_dict[k] /= total_chances  # sum of hypotheses = 1 (hopefully)

        return public_state_dict

    @property
    def public_belief_states_latter(self) -> Dict[RoleAllocationEnum, float]:
        """Public belief states(?) for this gamestate (after the outcome of this mission attempt)."""

        player_kv: List[Tuple[TPlayer, float]] = [*self._latter_predicted_spy_probabilities.items()]

        public_state_dict: Dict[RoleAllocationEnum, float] = {}

        for rp in RoleAllocationEnum.__members__.values():
            chance_of_this_allocation: float = 1
            rp_list = rp.get_value()
            for r in range(0, len(rp_list)):
                if rp_list[r]:
                    chance_of_this_allocation *= player_kv[r][1]
                else:
                    chance_of_this_allocation *= (1 - player_kv[r][1])
            public_state_dict[rp] = chance_of_this_allocation

        total_chances: float = sum(public_state_dict.values())

        if total_chances == 0:
            total_chances = 1

        for k in public_state_dict.keys():
            public_state_dict[k] /= total_chances  # sum of hypotheses = 1 (hopefully)

        return public_state_dict


    def json_dumpable_public_belief_states(self, latter: bool = False) -> Dict[str, float]:
        """
        Wrapper for self.public_belief_states_former and also public_belief_states_latter
        that returns them in a format that's more json-friendly.
        Why?
        Attempting a json.dumps on the public_belief_states causes a
        'TypeError: keys must be str, int, float, bool or None, not RoleAllocationEnum' error message.
        So I'm converting the keys to str instead.
        :param latter: set this to true if you want the latter public belief states (from after the mission outcome
        :return: dict with keys ["ab"] where a = index of spy 1, b = index of spy 2.
        """
        pbs: Dict[RoleAllocationEnum, float] = \
            self.public_belief_states_latter if latter else self.public_belief_states_prior

        jpbs: Dict[str, float] = {}

        for kv in pbs.items():
            jpbs[kv[0].to_string_for_json()] = kv[1]

        return jpbs

    @property
    def json_dumpable_individual_beliefs(self) -> tuple[float, float, float, float, float]:
        """
        gets prior_predicted_spy_probabilities, but in a tuple of floats (based on player index)
        :return: (p0 spy chance, p1 spy chance, p2 spy chance, p3 spy chance, p4 spy chance)
        """
        # noinspection PyTypeChecker
        return tuple([p for p in self._prior_predicted_spy_probabilities.values()])

    def __str__(self):
        """Formats this as a string, shamelessly lifted from the game.State class"""
        output: str = "<TeamRecord\n"
        for key in sorted(self.__dict__):
            value = self.__dict__[key]
            output += "\t- %s: %r\n" % (key, value)
        output += "\t- %s: %r\n" % ("loggable_dict", self.loggable_dict)
        pbs: Dict[RoleAllocationEnum, float] = self.public_belief_states_prior
        output += "\t- %s: %r\n" % ("public_belief_dict", pbs)
        output += "\t- %s: %r\n" % ("most_sus_pair", max(pbs.items(), key=lambda kv: kv[1]))
        output += "\t- %s: %r\n" % ("public_belief_dict_json", self.json_dumpable_public_belief_states())
        return output + ">"

    @property
    def loggable_dict(self) -> Dict[
        str,
        Union[
            Tuple[int, int, int, int, int],
            Dict[str, float],
            Tuple[float, float, float, float, float],
            int
        ]
    ]:
        """
        Attempts to turn this into a dict that can be logged
        :return: a dictionary with the following values:
        * leader
            * tuple of (leader index, leader suspicion)
            * one-hot encoded tuple of 4 0s and one 1 (with the 1 being in leader[self.leader.index])
        * beliefs_prior
            * a dict of
                * RoleAllocationEnum.to_string_for_json()
                * float chance of each RoleAllocationEnum describing which team are spies,
                calculated before this round's outcome
                    * normalized so the sum of all chances = 1
        * beliefs_latter
            * Same as above, but using the updated predictions from after the end of the round.
        * team
            * one-hot encoded tuple, indicating which members were on the team (1 = member with that index on team)
        * voted
            * one-hot encoded tuple, indicating which team members voted for the team (1 = member with that index on team)
        * sabotaged
            * how many times the mission was sabotaged (0 if success, -1 if nomination failed)

        """
        #info_dict: Dict[str, Union[float, Tuple[int, float], Dict[int, float], int]] = {}

        #prior_probs: List[float] = [*self._prior_predicted_spy_probabilities.values()]

        #for i in range(0, len(prior_probs)):
        #    info_dict["p{}".format(i)] = prior_probs[i]

        info_dict: Dict[str, Union[
            Tuple[int, int, int, int, int],
            Dict[str, float],
            Tuple[float, float, float, float, float],
            int]
        ] = {}

        default_leader_array: List[int, int, int, int, int] = [0, 0, 0, 0, 0]
        default_leader_array[self.leader.index] = 1

        # noinspection PyTypeChecker
        info_dict["leader"] = tuple(default_leader_array)

        #info_dict["leader"] = (self.leader.index, self._prior_predicted_spy_probabilities[self.leader])

        info_dict["beliefs_prior"] = self.json_dumpable_public_belief_states(False)

        info_dict["beliefs_latter"] = self.json_dumpable_public_belief_states(True)

        default_team_array: List[int, int, int, int, int] = [0, 0, 0, 0, 0]
        for p in self.team:
            default_team_array[p.index] = 1

        # noinspection PyTypeChecker
        info_dict["team"] = tuple(default_team_array)

        default_voted_array: List[int, int, int, int, int] = [0, 0, 0, 0, 0]
        for p in self.voted_for_team:
            default_voted_array[p.index] = 1

        # noinspection PyTypeChecker
        info_dict["voted"] = tuple(default_voted_array)

        #for p in self.team:
        #    info_dict["team"][p.index] = info_dict["p{}".format(p.index)]
        #{p: self._prior_predicted_spy_probabilities[p] for p in self.team}

        info_dict["sabotaged"] = self._sabotages

        # info_dict["players"] = self.json_dumpable_individual_beliefs

        return info_dict



class TempTeamRecord(object):
    """
    Like TeamRecord, but temporary.
    All fields are mutable so they can be updated during the round.
    Intended to be turned into a proper TeamRecord (with all the info in it) after the end of the round,
    so it can be added to the history of team records.
    """

    def __init__(self):
        self.team: Tuple[TPlayer, ...] = ()
        self.leader: TPlayer = None
        self.mission_number: int = 0
        self.nomination_attempt: int = 0
        self.nomination_successful: bool = False
        self._sabotages: int = -1
        self._voted_for_team: FrozenSet[TPlayer] = frozenset()
        self.prior_player_spy_probabilities: Dict[TPlayer, float] = {}
        self.latter_player_spy_probabilities: Dict[TPlayer, float] = {}

    def reset_at_round_start(self, ldr: TPlayer, mis: int, nom: int) -> None:
        """
        Call this at the start of round. Resets data, allowing this round's data to be copied in.
        :param ldr: current leader
        :param mis: mission number
        :param nom: nomination attempt
        """
        self.leader = ldr
        self.mission_number = mis
        self.nomination_attempt = nom

        self.team = ()
        self.nomination_successful = False
        self._sabotages = -1
        self._voted_for_team = ()

    def add_team_info(self, team: Iterable[TPlayer], prior_spy_probs: Dict[TPlayer, float]) -> None:
        """
        Copy team info to this object when the team is revealed
        :param team: the new team
        """
        self.team = tuple(team)
        self.prior_player_spy_probabilities = prior_spy_probs.copy()

    def add_vote_info(self, yes_men: Iterable[TPlayer], approved: bool) -> None:
        """
        Copies info about the vote to the team record
        :param yes_men: players who voted in favour of it
        :param approved: true if the vote passed
        """
        self._voted_for_team = frozenset(yes_men)
        self.nomination_successful = approved

    def add_mission_outcome_info(self, sab: int) -> None:
        """
        Adds the mission outcome info to the TeamRecord
        :param sab: number of times this mission was sabotaged
        """
        self._sabotages = sab

    def add_current_spy_probabilities(self, spy_probs_dict: Dict[TPlayer, float]) -> None:
        """
        Adds the current spy probabilities (worked out heuristically) to the team record info
        :param spy_probs_dict: relative probability of each player being spy (individually)

        :return: nothing
        """
        self.latter_player_spy_probabilities = spy_probs_dict.copy()
        #self.player_spy_probabilities: Dict[TPlayer, float] = {}
        #for p in [*spy_probs_dict.keys()]:
        #    self.player_spy_probabilities[p] = spy_probs_dict[p]


    @property
    def generate_teamrecord_from_data(self) -> TeamRecord:
        """
        Puts the data in this TempTeamRecord into a proper TeamRecord so it can be saved for later
        :return: a TeamRecord with the current data from this TempTeamRecord
        """
        return TeamRecord(self.team, self.leader, self.mission_number, self.nomination_attempt,
                          self.nomination_successful, self._sabotages, self._voted_for_team,
                          self.prior_player_spy_probabilities, self.latter_player_spy_probabilities)

    @property
    def sabotages(self) -> int:
        """sabotage count (-1: nomination failed. 0: mission pass. 1+: mission fail)"""
        return self._sabotages

    @property
    def voted_for_team(self) -> FrozenSet:
        """who voted for the team?"""
        return self._voted_for_team




class PlayerRecord(object):
    """
    A class to hold some data about suspicions of a single player.

    Contains lists with the missions that this player lead, teams they were on,
    and which teams they voted for/against, along with a dictionary which is a copy of
    the [mission state index, sabotage count] data for the missions
    (effectively being used as an ordered set of tuples).

    Please go to the GamestateTree if you wish to learn about the indexing.

    Anyway, there's also a buttload of properties on this class,
    which are basically just used to convert the aggregate lists into smaller lists,
    splitting the mission info into categories based on what the outcomes of them were
    (success, sabotaged, team rejected)

    """

    def __init__(self, p: TPlayer):
        """
        Constructor
        :param p: the player we're keeping an eye on
        """
        self._p: TPlayer = p
        """Who this player is"""

        self._all_missions_and_sabotages_with_teams: Dict[int, Tuple[int, int]] = {}
        """
        Dictionary of mission IDs with sabotage counts.
        (ID, sabotage count)
        Missions identified via GameState_Tree indices
        if sabotage is -1, that means the team was rejected.
        """

        self._missions_lead: List[int] = []
        """
        The missions they have lead, identified via GameState_Tree indices.
        """

        self._teams_been_on: List[int] = []
        """
        The teams that this player has been on, identified via GameState_Tree indices.
        """

        self._teams_approved: List[int] = []
        """
        The teams that this player voted for, identified via GameState_Tree indices
        """

        self._teams_rejected: List[int] = []
        """
        The teams that this player voted against,identified via GameState_Tree indices.
        """

        self._is_spy: bool = None
        """IMPORTANT: DO NOT GIVE THIS A VALUE UNTIL KNOWN, FOR SURE, WHETHER THIS PLAYER WAS A SPY"""

    def post_round_update(self, index: int, sab: int, team_size: int, was_leader: bool,
                          was_on_team: bool, voted_for_team: bool) -> None:
        """
        Call this to update the player info with the data for each round.
        :param index: mission ID (via GamestateTree indices)
        :param team_size: how many members were on the team
        :param sab: sabotage count (-1 if team rejected)
        :param was_leader: true if this player was the leader of the team.
        :param was_on_team: true if this player was leading this team.
        :param voted_for_team: true if this player voted for the team.
        :return: nothing.
        """

        self._all_missions_and_sabotages_with_teams[index] = (sab, team_size)

        if was_leader:
            self._missions_lead.append(index)

        if was_on_team:
            self._teams_been_on.append(index)

        if voted_for_team:
            self._teams_approved.append(index)
        else:
            self._teams_rejected.append(index)

        pass

    def identity_is_known(self, actually_is_spy: bool) -> None:
        """
        Call this when we know for sure what role this player had.
        :param actually_is_spy: true if they actually are a spy, false otherwise.
        :return: nothing
        """
        self._is_spy = actually_is_spy

    # got so many properties here its looking like a monopoly board

    @property
    def p(self) -> TPlayer:
        """The player themselves."""
        return self._p

    @property
    def pname(self) -> str:
        """The name of the player this stats object is recording data for"""
        return self._p.name

    @property
    def is_spy(self) -> bool:
        """Whether or not this player is a spy. Returns None if not known for sure yet."""
        return self._is_spy

    @property
    def all_missions_lead(self) -> List[int]:
        """All missions that this player lead"""
        return self._missions_lead.copy()

    @property
    def passed_missions_lead(self) -> List[int]:
        """All passed missions that this player lead"""
        return [m for m in self._missions_lead if self._all_missions_and_sabotages_with_teams[m][0] == 0]

    @property
    def failed_missions_lead(self) -> List[int]:
        """All failed missions that this player lead"""
        return [m for m in self._missions_lead if self._all_missions_and_sabotages_with_teams[m][0] > 0]


    @property
    def mission_teams_lead_sus_levels(self) -> List[Tuple[int, float]]:
        """
        finds all non-rejected teams lead, and returns a list of [id for that gamestate, sabotages/participants].
        Grants leniency for teams that were used for the first mission (gamestates 0 to -4).
        """
        teams_sus: List[Tuple[int, float]] = []
        for t in self._missions_lead:
            if 0 >= t > -5: # mission 1 teams omitted
                continue
            sabs: Tuple[int, int] = self._all_missions_and_sabotages_with_teams[t]
            if sabs[0] == -1:
                continue
            elif sabs[0] == 0:
                teams_sus.append((t, sabs[0]))
            else:
                teams_sus.append((t, sabs[0]/sabs[1]))
        return teams_sus

    @property
    def rejected_missions_lead(self) -> List[int]:
        """All rejected teams that this player nominated"""
        return [m for m in self._missions_lead if self._all_missions_and_sabotages_with_teams[m][0] == -1]

    @property
    def teams_been_on(self) -> List[int]:
        """All teams that this player has been on"""
        return self._teams_been_on.copy()

    @property
    def passed_teams_been_on(self) -> List[int]:
        """All passed missions that this player was on"""
        return [m for m in self._teams_been_on if self._all_missions_and_sabotages_with_teams[m][0] == 0]

    @property
    def failed_teams_been_on(self) -> List[int]:
        """All failed missions that this player was on"""
        return [m for m in self._teams_been_on if self._all_missions_and_sabotages_with_teams[m][0] > 0]

    @property
    def not_rejected_teams_been_on_with_sabotage_count_and_size(self) -> List[Tuple[int, int, int]]:
        """All non-rejected teams that this player was on,
        along with their sabotage count and teamsize) info"""
        the_teams_and_sab_counts: List[Tuple[int, int, int]] = []
        for t in self._teams_been_on:
            sabs: Tuple[int, int] = self._all_missions_and_sabotages_with_teams[t]
            if sabs[0] < 0: # we don't care about rejected teams.
                continue
            else:
                the_teams_and_sab_counts.append((t, sabs[0], sabs[1]))
        return the_teams_and_sab_counts

    @property
    def mission_teams_been_on_sus_levels(self) -> List[Tuple[int, float]]:
        """finds all non-rejected teams been on, and returns a list of [id for that gamestate, sabotages/participants]"""
        teams_sus: List[Tuple[int, float]] = []
        for t in self._teams_been_on:
            sabs: Tuple[int, int] = self._all_missions_and_sabotages_with_teams[t]
            if sabs[0] == -1:
                continue
            elif sabs[0] == 0:
                teams_sus.append((t, sabs[0]))
            else:
                teams_sus.append((t, sabs[0]/sabs[1]))
        return teams_sus

    @property
    def rejected_teams_been_on(self) -> List[int]:
        """All rejected teams that this player was on"""
        return [m for m in self._teams_been_on if self._all_missions_and_sabotages_with_teams[m][0] == -1]

    @property
    def teams_approved(self) -> List[int]:
        """All teams that this player voted for"""
        return self._teams_approved.copy()

    @property
    def passed_teams_approved(self) -> List[int]:
        """All passed missions that this player voted for"""
        return [m for m in self._teams_approved if self._all_missions_and_sabotages_with_teams[m][0] == 0]

    @property
    def failed_teams_approved(self) -> List[int]:
        """All failed missions that this player voted for"""
        return [m for m in self._teams_approved if self._all_missions_and_sabotages_with_teams[m][0] > 0]

    @property
    def rejected_teams_approved(self) -> List[int]:
        """All rejected teams that this player voted for"""
        return [m for m in self._teams_approved if self._all_missions_and_sabotages_with_teams[m][0] == -1]

    @property
    def approved_teams_sus_levels(self) -> List[Tuple[int, float]]:
        """Relative suspicion from teams that this player voted for"""
        teams_sus: List[Tuple[int, float]] = []
        for t in self._teams_approved:
            sabs: Tuple[int, int] = self._all_missions_and_sabotages_with_teams[t]
            if sabs[0] == -1:
                continue
            elif sabs[0] == 0:
                teams_sus.append((t, sabs[0]))
            else:
                teams_sus.append((t, sabs[0] / sabs[1]))
        return teams_sus

    @property
    def sus_levels_of_non_hammer_teams_approved_whilst_not_on(self) -> List[Tuple[int, float]]:
        """
        Relative suspicions from teams that this player voted for, whilst not being on aforementioned team.
        Grants leniency for hammer teams (as resistance members are forced to vote in favour of hammer teams),
        and votes for round 1 (as there's no real information for those votes)
        """
        approved_sus: List[Tuple[int, float]] = []
        hammers: List[int] = GamestateTree.get_hammer_indices()
        for t in self._teams_approved:
            if t in hammers:
                continue
            if 0 >= t > -5:
                continue
            sabs: Tuple[int, int] = self._all_missions_and_sabotages_with_teams[t]
            if sabs[0] == -1:
                continue
            elif sabs[0] == 0:
                continue
                #approved_sus.append((t, 0))
            else:
                approved_sus.append((t, sabs[0]/sabs[1]))
        return approved_sus

    @property
    def teams_rejected(self) -> List[int]:
        """All teams that this player voted against"""
        return self._teams_rejected.copy()

    @property
    def passed_teams_rejected(self) -> List[int]:
        """All passed missions that this player voted against"""
        return [m for m in self._teams_rejected if self._all_missions_and_sabotages_with_teams[m][0] == 0]

    @property
    def failed_teams_rejected(self) -> List[int]:
        """All failed missions that this player voted against"""
        return [m for m in self._teams_rejected if self._all_missions_and_sabotages_with_teams[m][0] > 0]

    @property
    def rejected_teams_rejected(self) -> List[int]:
        """All rejected teams that this player voted against"""
        return [m for m in self._teams_rejected if self._all_missions_and_sabotages_with_teams[m][0] == -1]

    @property
    def rejected_teams_sus_levels(self) -> List[Tuple[int, float]]:
        """Relative suspicion from teams that this player voted against"""
        teams_sus: List[Tuple[int, float]] = []
        for t in self._teams_rejected:
            sabs = self._all_missions_and_sabotages_with_teams[t]
            if sabs[0] == -1:
                continue
            elif sabs[0] == 0:
                teams_sus.append((t, sabs[0]))
            else:
                teams_sus.append((t, sabs[0] / sabs[1]))
        return teams_sus

    @property
    def hammer_votes(self) -> List[Tuple[int, bool]]:
        """How this player voted for the final nomination attempts"""
        hammer: List[int] = GamestateTree.get_hammer_indices()
        votes: List[Tuple[int, bool]] = []
        for v in self._teams_approved:
            if v in hammer:
                votes.append((v, True))
        for v in self._teams_rejected:
            if v in hammer:
                votes.append((v, False))
        return votes

    @property
    def hammers_thrown(self) -> List[int]:
        """How many hammers this player threw (final attempts rejected)"""
        hammer: List[int] = GamestateTree.get_hammer_indices()
        return [v for v in self._teams_rejected if v in hammer]

    def simple_spy_probability(self, everything_before_round: int = -1) -> float:
        """
        Works out how likely this player is to be a spy, given their actions until the current round.
        :param everything_before_round: Current round (rounds since game start). Defaults to -1
        If negative/unspecified/not an index of a round that has been reached, takes everything
        known so far into account. If specified (value of n), takes everything up before the nth round into account.

        :return: a float indicating how suspicious this player currently is.
        If there's no data yet, their sus level is 0.5
        If they have rejected a hammer vote, their sus level is 1
        otherwise,
        ((sabotages on lead missions * lead missions) + (sabotages on attended missions * attended)) / (lead * attended)
        """
        #import traceback
        #traceback.print_stack()

        # TODO more refined calculations of suspiciousness?
        #  Could try to use some neural networks/bayesian belief stuff/etc.
        #  maybe factoring in votes?


        if everything_before_round < 0:  # if negative value given, we take everything so far into account
            everything_before_round = len(self._all_missions_and_sabotages_with_teams)

        #print([*self._all_missions_and_sabotages_with_teams.keys()])
        #print([*self._all_missions_and_sabotages_with_teams.keys()][0:everything_before_round])

        all_prior_rounds: List[int] = [*self._all_missions_and_sabotages_with_teams.keys()][0:everything_before_round]

        #print(all_prior_rounds)

        if len(all_prior_rounds) == 0:
            return 0.25  # 0.25 suspicion if no data

        if len([h for h in self.hammers_thrown if h in all_prior_rounds]) > 0:
            return 1  # only a spy would throw a hammer vote.

        lead_sus: List[float] = [m[1] for m in self.mission_teams_lead_sus_levels if m[0] in all_prior_rounds]
        been_on_sus: List[float] = [m[1] for m in self.mission_teams_been_on_sus_levels if m[0] in all_prior_rounds]

        not_on_voted_sus: List[float] =\
            [m[1] for m in self.sus_levels_of_non_hammer_teams_approved_whilst_not_on if m[0] in all_prior_rounds]

        has_lead_missions: bool = len(lead_sus) > 0
        has_been_on_missions: bool = len(been_on_sus) > 0
        voted_for_missions_not_on: bool = len(not_on_voted_sus) > 0

        # voted_for_sus: List[float] = [m[1] for m in self.approved_teams_sus_levels if m[0] in all_prior_rounds]
        # voted_against_sus: List[float] = [m[1] for m in self.rejected_teams_sus_levels if m[0] in all_prior_rounds]

        # total_concluded_missions_voted_on: int = len(voted_for_sus) + len(voted_against_sus)

        if has_lead_missions or has_been_on_missions or voted_for_missions_not_on:
            lead_len = len(lead_sus)
            been_len = len(been_on_sus)
            voted_len = len(not_on_voted_sus)
            return min((sum(lead_sus)/4) + (sum(been_on_sus)) + (sum(not_on_voted_sus)/4) / (lead_len/4 + been_len + voted_len/4), 1)
            #  return (sum(lead_sus) * max(1,lead_len)) + (sum(been_on_sus) * max(1, been_len)) / (lead_len + been_len) * (max(1, lead_len) * max(1, been_len))
        else:
            return 0.25
        #elif total_concluded_missions_voted_on > 0:
        #    for_against_sus: float = max(
        #        (
        #                (sum(voted_for_sus) * len(voted_for_sus)) - (sum(voted_against_sus) * len(voted_against_sus))
        #        ) / total_concluded_missions_voted_on, 0)
        #    return for_against_sus #  0.25

    @property
    def get_info_about_approved_missions_this_player_was_on(self) -> List[Tuple[int, int, int, bool]]:
        """
        Gets info about the approved missions that this player was on.
        Returned in the form of a list of tuples (int, int, bool),
        where (mission id, sabotage count, was leader)
        Omits all missions with a sabotage count of -1 (not present in)
        :return: list of (mission id, sabotage count, sabCount/teamSize, was leader)
        for all non-rejected teams this player was in.
        """
        teams_in: List[Tuple[int, int, int]] = self.not_rejected_teams_been_on_with_sabotage_count_and_size
        return [(t[0], t[1], t[2], t[0] in self._missions_lead) for t in teams_in]


class SpySabotageChanceStats(object):
    """
    A class that holds the info about spy sabotage chances observed so far in this
    tournament. Not really viable to save this stuff in the logs, as this saves stuff
    based on player name, and chances are there might be completely different player
    names in each tournament (and a potentially infinite total number of players),
    so attempting to record this forever could go badly.
    """

    class IndividualSpySabStats(object):
        """
        An inner class that keeps a record of the stats for a specific spy.
        """

        COUNT: str = "count"
        SABOTAGED: str = "sabotaged"
        LEADER: str = "leader"
        NOT_LEADER: str = "not leader"
        AWKWARD: str = "awkward"

        def __init__(self, name: str):
            """
            Constructor. Records what this spy is called, and starts recording some stats.
            :param name: the name of this spy.
            """
            self._name: str = name

            ntgi: List[int] = GamestateTree.get_all_non_terminal_gamestates_indices()

            self._lead_missions_at_gamestates: Dict[int, Dict[str, int]] = dict.fromkeys(
                ntgi,
                {
                    SpySabotageChanceStats.IndividualSpySabStats.COUNT: 0,
                    SpySabotageChanceStats.IndividualSpySabStats.SABOTAGED: 0,
                }
            )
            """
            For each gamestate, keeps a count of how many times it lead a non-rejected mission
            (where it was the only spy)
            and how many times it sabotaged the aforementioned mission
            """

            self._been_on_missions_at_gamestates: Dict[int, Dict[str, int]] = dict.fromkeys(
                ntgi,
                {
                    SpySabotageChanceStats.IndividualSpySabStats.COUNT: 0,
                    SpySabotageChanceStats.IndividualSpySabStats.SABOTAGED: 0,
                }
            )
            """
            For each gamestate, keeps a count of how many times it was on a non-rejected mission
            (where it was the only spy)
            and how many times it sabotaged aforementioned missions.
            """

            self._missions_with_other_spy_at_gamestates: Dict[
                int, Dict[str, Dict[str, Union[int, float]]]
            ] = dict.fromkeys(
                ntgi,
                {
                    SpySabotageChanceStats.IndividualSpySabStats.LEADER: {  # sabotages when this spy is leading
                        SpySabotageChanceStats.IndividualSpySabStats.COUNT: 0,
                        SpySabotageChanceStats.IndividualSpySabStats.SABOTAGED: 0
                    },
                    SpySabotageChanceStats.IndividualSpySabStats.NOT_LEADER: {  # sabotages when other spy is leading
                        SpySabotageChanceStats.IndividualSpySabStats.COUNT: 0,
                        SpySabotageChanceStats.IndividualSpySabStats.SABOTAGED: 0
                    },
                    SpySabotageChanceStats.IndividualSpySabStats.AWKWARD: {  # sabotages when not spy is leading
                        SpySabotageChanceStats.IndividualSpySabStats.COUNT: 0,
                        SpySabotageChanceStats.IndividualSpySabStats.SABOTAGED: 0
                    }
                }
            )
            """
            For each gamestate, keeps track of all the times it was on a mission with another spy, in the form of
            a list with the sabotage counts for those missions.
            [i]["leader"] is a list of the sabotage counts for every two-spy mission this spy lead.
            [i]["accomplice"] is a list of the sabotage counts for every two-spy missions that the other spy lead
            [i]["idiot"] is a list of sabotage counts for every two-spy missions lead by a resistance member.
            """
            self._the_general_case: Dict[str, Dict[str, Union[int, Dict[str, float]]]] = {
                SpySabotageChanceStats.IndividualSpySabStats.LEADER: {
                    SpySabotageChanceStats.IndividualSpySabStats.COUNT: 0,
                    SpySabotageChanceStats.IndividualSpySabStats.SABOTAGED: 0
                },
                SpySabotageChanceStats.IndividualSpySabStats.NOT_LEADER: {
                    SpySabotageChanceStats.IndividualSpySabStats.COUNT: 0,
                    SpySabotageChanceStats.IndividualSpySabStats.SABOTAGED: 0
                },
                SpySabotageChanceStats.IndividualSpySabStats.AWKWARD: {
                    SpySabotageChanceStats.IndividualSpySabStats.LEADER: {
                        SpySabotageChanceStats.IndividualSpySabStats.COUNT: 0,
                        SpySabotageChanceStats.IndividualSpySabStats.SABOTAGED: 0
                    },
                    SpySabotageChanceStats.IndividualSpySabStats.NOT_LEADER: {
                        SpySabotageChanceStats.IndividualSpySabStats.COUNT: 0,
                        SpySabotageChanceStats.IndividualSpySabStats.SABOTAGED: 0
                    },
                    SpySabotageChanceStats.IndividualSpySabStats.AWKWARD: {
                        SpySabotageChanceStats.IndividualSpySabStats.COUNT: 0,
                        SpySabotageChanceStats.IndividualSpySabStats.SABOTAGED: 0
                    }
                }
            }
            """
            And the general case (aggregate for all gamestates).
            Single spy goes in the leader/not leader counts.
            Multiple spies goes in the awkward sub-dictionary
            """

        def _add_info_for_multiple_spies(self, sab_count: float, gs_id: int, condition: str) -> None:
            """
            Adds info to the dictionaries for when there are multiple spies
            :param sab_count: sabotage count (after being divided by 2)
            :param gs_id: id for the gamestate
            :param condition: either LEADER, NOT_LEADER, or AWKWARD
            """

            self._missions_with_other_spy_at_gamestates[gs_id][condition]\
                [SpySabotageChanceStats.IndividualSpySabStats.COUNT] += 1
            self._missions_with_other_spy_at_gamestates[gs_id][condition]\
                [SpySabotageChanceStats.IndividualSpySabStats.SABOTAGED] += sab_count

            self._the_general_case[SpySabotageChanceStats.IndividualSpySabStats.AWKWARD][condition]\
                [SpySabotageChanceStats.IndividualSpySabStats.COUNT] += 1
            self._the_general_case[SpySabotageChanceStats.IndividualSpySabStats.AWKWARD][condition] \
                [SpySabotageChanceStats.IndividualSpySabStats.SABOTAGED] += sab_count

        def _add_info_for_general_case_one_spy(self, sab: int, condition: str) -> None:
            """
            Adds info for the general case for when there's only one spy
            :param sab: sabotage count
            :param condition: leader/not leader
            """
            self._the_general_case[condition][SpySabotageChanceStats.IndividualSpySabStats.COUNT] += 1
            self._the_general_case[condition][SpySabotageChanceStats.IndividualSpySabStats.SABOTAGED] += sab

        def add_data_from_game(self, spy_info_tuples: List[Tuple[int, int, bool, bool, bool]]) -> None:
            """
            Adds the data about the spies for the given game to the
            :param spy_info_tuples: (mission id, sabotage count, isLeader, multipleSpies, leadByOtherSpy)
            """

            for t in spy_info_tuples:
                gs_id: int = t[0]
                sab_count: int = t[1]
                if t[3]:  # if multiple spies
                    f_sabs: float = sab_count/2
                    if t[2]:  # if lead by self
                        self._add_info_for_multiple_spies(
                            f_sabs, gs_id, SpySabotageChanceStats.IndividualSpySabStats.LEADER
                        )
                    elif t[4]:  # if lead by other spy
                        self._add_info_for_multiple_spies(
                            f_sabs, gs_id, SpySabotageChanceStats.IndividualSpySabStats.NOT_LEADER
                        )
                    else:  # if lead by non-spy
                        self._add_info_for_multiple_spies(
                            f_sabs, gs_id, SpySabotageChanceStats.IndividualSpySabStats.AWKWARD
                        )
                elif t[2]:  # if lead by this spy
                    self._lead_missions_at_gamestates[gs_id][SpySabotageChanceStats.IndividualSpySabStats.COUNT] += 1
                    self._lead_missions_at_gamestates[gs_id][SpySabotageChanceStats.IndividualSpySabStats.SABOTAGED]\
                        += sab_count
                    self._add_info_for_general_case_one_spy(
                        sab_count, SpySabotageChanceStats.IndividualSpySabStats.LEADER
                    )
                else:  # if lead by non-spy
                    self._been_on_missions_at_gamestates[gs_id][SpySabotageChanceStats.IndividualSpySabStats.COUNT] += 1
                    self._been_on_missions_at_gamestates[gs_id][SpySabotageChanceStats.IndividualSpySabStats.SABOTAGED]\
                        += sab_count
                    self._add_info_for_general_case_one_spy(
                        sab_count, SpySabotageChanceStats.IndividualSpySabStats.NOT_LEADER
                    )

        def _multiple_spy_probability(self, gs: int, condition: str) -> float:
            """
            Obtains probability from multiple spies for the given gamestate, or for the general case,
            with the given condition
            :param gs: gamestate ID
            :param condition: leader/not_leader/awkward
            :return: probability of this spy sabotaging in that gamestate, or in the general case, or -1 if no
            general case examples found
            """
            count_a: int = self._missions_with_other_spy_at_gamestates[gs][condition]\
                [SpySabotageChanceStats.IndividualSpySabStats.COUNT]
            if count_a > 0:
                return self._missions_with_other_spy_at_gamestates[gs][condition]\
                           [SpySabotageChanceStats.IndividualSpySabStats.SABOTAGED] / count_a

            gc_count: float = self._the_general_case[SpySabotageChanceStats.IndividualSpySabStats.AWKWARD][condition]\
                [SpySabotageChanceStats.IndividualSpySabStats.COUNT]
            if gc_count > 0:
                return self._the_general_case[SpySabotageChanceStats.IndividualSpySabStats.AWKWARD][condition]\
                [SpySabotageChanceStats.IndividualSpySabStats.SABOTAGED] / gc_count
            else:
                return -1

        def _single_spy_general_case_probability(self, condition: str) -> float:
            """
            Obtains probability from general case for a single spy under the given condition
            :param condition: leader/not_leader
            :return: probability of this spy sabotaging under that condition in the general case, or -1 if no
            general case examples found
            """
            gc_count: int = self._the_general_case[condition][SpySabotageChanceStats.IndividualSpySabStats.COUNT]
            if gc_count > 0:
                return self._the_general_case[condition][SpySabotageChanceStats.IndividualSpySabStats.SABOTAGED] / \
                       gc_count
            else:
                return -1

        def get_sabotage_chance(self, gs: int, is_leader: bool, multiple_spies: bool, leader_is_spy: bool):
            """
            Attempts to get the likelihood of this spy sabotaging stuff in this situation at the given gamestate.
            Returns -1 if no data can be found for that gamestate.
            :param gs: the current gamestate
            :param is_leader: is this spy the leader?
            :param multiple_spies: are there multiple spies on the team?
            :param leader_is_spy: is the spy a leader (if there are multiple spies)?
            :return: probability (0-1) of the spy sabotaging stuff in this situation at the given gamestate.
            If no data for that particular situation at the gamestate could be found, returns the probability
            for the general case at that gamestate.
            If no data could be found for the general case at this gamestate, returns -1.
            """
            if multiple_spies:
                # For cases where there are multiple spies, we divide the sum of sabotage counts for that situation by
                # twice the length of the list of sabotages for that situation, because individual sab counts can
                # be either 0, 1, or 2, and we need a result between 0-1.
                if is_leader:
                    return self._multiple_spy_probability(gs, SpySabotageChanceStats.IndividualSpySabStats.LEADER)
                elif leader_is_spy:
                    return self._multiple_spy_probability(gs, SpySabotageChanceStats.IndividualSpySabStats.NOT_LEADER)
                else:
                    return self._multiple_spy_probability(gs, SpySabotageChanceStats.IndividualSpySabStats.AWKWARD)
            elif is_leader:
                lead_count: int = self._lead_missions_at_gamestates[gs]\
                                        [SpySabotageChanceStats.IndividualSpySabStats.COUNT]
                if lead_count > 0:
                    return self._lead_missions_at_gamestates[gs]\
                               [SpySabotageChanceStats.IndividualSpySabStats.SABOTAGED] / lead_count
                else:
                    return self._single_spy_general_case_probability(SpySabotageChanceStats.IndividualSpySabStats.LEADER)
            else:
                not_lead_count: int = self._been_on_missions_at_gamestates[gs]\
                    [SpySabotageChanceStats.IndividualSpySabStats.COUNT]
                if not_lead_count > 0:
                    return self._been_on_missions_at_gamestates[gs]\
                               [SpySabotageChanceStats.IndividualSpySabStats.SABOTAGED] / not_lead_count
                else:
                    return self._single_spy_general_case_probability(
                        SpySabotageChanceStats.IndividualSpySabStats.NOT_LEADER
                    )

    def __init__(self):
        # TODO: set this up, dictionary of spy stat objects, and one dummy stat object as an aggregate for agents without
        #  any of their own data
        # TODO: how likely is a given spy to sabotage?
        self._default_stats: "SpySabotageChanceStats.IndividualSpySabStats" =\
            SpySabotageChanceStats.IndividualSpySabStats("default")
        """A stats object for a generic spy (aggregate for every spy ever)"""

        self._player_stats: Dict[str, "SpySabotageChanceStats.IndividualSpySabStats"] = {}
        """A dictionary to keep the stats for every spy encountered in this play session.
        This is only being stored in RAM instead of being stored on disk/used in some neural stuff,
        because, over time, there might be a lot of spies, who may have non-unique names,
        and that could get out of hand pretty quickly, y'know?"""

        pass

    def add_sabotage_info(self, sabotage_info_dict: Dict[str, List[Tuple[int, int, bool, bool, bool]]]) -> None:
        """
        Adds the sabotage info from a gamerecord to the sabotage info stats for the specified spies (and the
        generic spy)
        :param sabotage_info_dict: the dictionary returned from the GameRecord's
        get_info_about_sabotages_from_spies_for_sabotage_records method
        :return: nothing
        """
        for kv in sabotage_info_dict.items():

            if kv[0] not in self._player_stats:
                self._player_stats[kv[0]] = SpySabotageChanceStats.IndividualSpySabStats(kv[0])
            self._player_stats[kv[0]].add_data_from_game(kv[1])
            self._default_stats.add_data_from_game(kv[1])
        pass

    def get_sabotage_chance(
            self,
            spy_name: str,
            gs: int,
            is_leader: bool,
            multiple_spies: bool,
            leader_is_spy: bool
    ) -> float:
        """
        Attempts to work out how likely this spy is to sabotage, from all the spy sabotage info we have so far.
        :param spy_name: the name of this spy.
        :param gs: id of the current gamestate
        :param is_leader: is this spy leading the team?
        :param multiple_spies: are there multiple spies on the team?
        :param leader_is_spy: is the leader of this team a spy?
        :return: the probability of this spy attempting to sabotage in this situation, from prior observations.
        """
        # if we have data for this spy, we attempt to get data from their stats object
        if spy_name in self._player_stats:
            individual_probs: float = self._player_stats[spy_name].get_sabotage_chance(
                gs, is_leader, multiple_spies,leader_is_spy
            )
            if individual_probs != -1:
                return individual_probs

        # if we don't have data for that spy, or if their data returns -1, we get data for the general case instead.
        general_probs: float = self._default_stats.get_sabotage_chance(
            gs, is_leader, multiple_spies, leader_is_spy
        )
        if general_probs == -1:  # if we get nothing for the general case, we give up and return 0.5
            return 0.5
        else:
            return general_probs  # otherwise we return what we got for the general case.


class GameRecord(object):
    """
    A class used to hold data about an entire game,
    before getting it in JSON loggable form,
    also making it easier to update the SpySabotageChanceStats object
    after the end of the game.
    """

    def __init__(self):
        self._team_records: Dict[int, TeamRecord] = {}
        """
        A record of what teams have been nominated so far,
        who nominated the team,
        the results of those votes (who voted in favour of it),
        and what the outcomes of those missions were (if nomination succeeded).
        Key values correspond to the index of the relevant GamestateTree node in the GamestateTree object.
        The python specification states that dict.keys() is ALWAYS ordered by insertion order, so this also
        keeps track of a gamestate history, which is nice as well.
        """

    def add_teamrecord_to_records(self, gs: int, tr: TeamRecord) -> None:
        """
        Adds the given teamrecord to this gamerecord at the given gamestate.
        :param gs: the id of the gamestate (gamestatetree ID)
        :param tr: the actual teamrecord for that gamestate
        :return: nothing.
        """
        self._team_records[gs] = tr

    def reset(self) -> None:
        """
        Clears everything from team_records, so we can start a new game.
        :return:
        """
        self._team_records.clear()

    @property
    def json_loggable_teamrecords(self) -> Dict[
        int,
        Union[
            Tuple[int, int, int, int, int],
            Dict[str, float],
            Tuple[float, float, float, float, float],
            int
        ]
    ]:
        """
        Returns a dictionary of the loggable teamrecord dicts,
        for use when logging json stuff
        :return: a dictionary of all the loggabledicts of all the teamrecords
        """
        log_dict: Dict[
            int,
            Union[
                Tuple[int, int, int, int, int],
                Dict[str, float],
                Tuple[float, float, float, float, float],
                int
            ]
        ] = {}
        for kv in self._team_records.items():
            log_dict[kv[0]] = kv[1].loggable_dict

        return log_dict

    def get_info_about_sabotages_from_spies_for_sabotage_records(self,
         the_spies: List[TPlayer],
         me: TPlayer,
         im_spy: bool,
         my_known_sabs: List[int]
    ) -> Dict[
        str, List[Tuple[int, int, bool, bool, bool]]
    ]:
        """
        Obtains info about spy sabotages from the gamerecords, for use in the spysabotagerecords
        :param the_spies: A tuple of who the spies were.
        :param me: the player who is calling this
        :param im_spy: if the player who is calling this is a spy
        :param my_known_sabs: If caller was a spy, this holds the list of all the missions that they definitely did
        sabotage, so, for missions where both spies were present, this allows us to see exactly how many times
        the other spy sabotaged it.
        :return: A dictionary of [spy name, list(mission id, sabotage count, isLeader, multipleSpies, leadBySpy)]
        for both of those spies,
        which can be easily fed to a SpySabotageChances object.
        """
        spy1_name: str = the_spies[0].name
        spy2_name: str = the_spies[1].name
        info_dict: Dict[str, List[Tuple[int, int, bool, bool, bool]]] = {
            spy1_name: [],
            spy2_name: []
        }
        im_spy1: bool = im_spy and me == the_spies[0]
        im_spy2: bool = im_spy and not im_spy1

        missions_that_happened: List[Tuple[int, TeamRecord]] = [kv for kv in self._team_records.items() if
                                                                kv[1].sabotages != -1]
        for m in missions_that_happened:

            tr: TeamRecord = m[1]

            team: Tuple[TPlayer] = tr.team
            spy_1_on_team: bool = the_spies[0] in team
            spy_2_on_team: bool = the_spies[1] in team

            if spy_1_on_team or spy_2_on_team:
                leader_is_spy: bool = tr.leader in the_spies
                spy_1_is_leader: bool = leader_is_spy and tr.leader == the_spies[0]
                spy_2_is_leader: bool = leader_is_spy and not spy_1_is_leader
                multiple_spies: bool = spy_1_on_team and spy_2_on_team
                sabs1: int = tr.sabotages
                sabs2: int = sabs1
                if im_spy and multiple_spies and sabs1 == 1:
                    # if there were multiple spies, but only one sabotage, and I'm a spy
                    if im_spy1 ^ m[0] in my_known_sabs:
                        # If I either did it as spy 2, or didn't do it as spy 1,
                        # we know that spy 1 definitely didn't do it, and that
                        # spy 2 definitely did, so we consider spy 2 to have sabotaged
                        # it twice, whilst spy 1 didn't sabotage it at all.
                        sabs1 = 0
                        sabs2 = 2
                    else:  # and vice versa.
                        sabs1 = 2
                        sabs2 = 0

                if spy_1_on_team:
                    info_dict[spy1_name].append(
                        (m[0], sabs1, spy_1_is_leader, multiple_spies, leader_is_spy)
                    )
                if spy_2_on_team:
                    info_dict[spy2_name].append(
                        (m[0], sabs2, spy_2_is_leader, multiple_spies, leader_is_spy)
                    )

        return info_dict


class NeuralNetworker(object):
    """
    A class that loads and holds all of the neural networks used by this bot,
    as well as providing wrappers for using the neural networks.
    """

    def __init__(self):
        # TODO: methods to load the neural networks
        pass


class rl18730(Bot):
    """
    Rachel's somewhat functional bot for playing The Resistance.

    note: currently not functional.
    """

    _sabotage_chance_stats: SpySabotageChanceStats = SpySabotageChanceStats()
    """
    This is a SpySabotageChanceStats object, to keep track of the sabotage chances for every single spy
    that we encounter.
    
    This is being done as a static attribute belonging to the class itself, instead of being initialized in init,
    because this framework appears to create a new instance of this bot whenever it's being included in a game,
    so, if this object only belonged to an instance of this class, it would be reset every single game,
    making it incredibly useless. So it's here instead.
    """
    if os.path.exists(resources_file_path / "sabotages.p"):
        with open(resources_file_path / "sabotages.p", "rb") as p:
            _sabotage_chance_stats: SpySabotageChanceStats = pickle.load(p)
            p.close()

    def __init__(self, game: State, index: int, spy: bool):
        """Constructor called before a game starts.  It's recommended you don't
        override this function and instead use onGameRevealed() to perform
        setup for your AI.
        :param game:     the current game state
        :param index:    Bot's index in the player list.
        :param spy:      Is this bot meant to be a spy?

        Note to self: init appears to be called at the start of every game that this bot is in.

        First step: probably trying to get some sort of bayesian classifier working.
        10 possible role combinations (6 of which need to be worked out if resistance),
        need to think of how to store a bayesian belief network into a data structure,
        and how to get code to use it.

        Might need to take another look at the statistician bot, because that appears to use a naive bayes classifier.
        Or look at exising Python implementations of it for later on.

        """
        super().__init__(game, index, spy)

        self.spies: List[TPlayer] = []
        """List of known spies (empty unless spy)"""

        self.current_gamestate: int = 0
        """Will hold the index of the current gamestate (via GamestateTree)"""

        self.game_record: GameRecord = GameRecord()
        """
        A record of the info for this game.
        """

        self.team_records: Dict[int, TeamRecord] = {}
        """
        A record of what teams have been nominated so far,
        who nominated the team,
        the results of those votes (who voted in favour of it),
        and what the outcomes of those missions were (if nomination succeeded).
        Key values correspond to the index of the relevant GamestateTree node in the GamestateTree object.
        The python specification states that dict.keys() is ALWAYS ordered by insertion order, so this also
        keeps track of a gamestate history, which is nice as well.
        """

        self.temp_team_record: TempTeamRecord = TempTeamRecord()
        """The temporary teamrecord, which will hold info for the current team, and will be updated on the fly,
        beofre eventually being added to the history of team records."""

        self.player_records: Dict[TPlayer, PlayerRecord] = {}
        """
        A dictionary to hold PlayerRecord objects for all the individual players.
        """

        self.leader_order: List[TPlayer] = []
        """Keeps track of the order of leaders (0->1st leader, 4->final (has hammer))"""

        self.suspicion_for_each_role_combo: Dict[RoleAllocationEnum, float] = \
            dict.fromkeys(RoleAllocationEnum.__members__.values(), 0.1)
        """
        Relative probabilities of each possible assignment of roles (as of right now).
        
        Keys: RoleAllocationEnum 
        Values: (p1 spy chance + p2 spy chance)/2 -> chance that those two players are the spies.
        """
        self.role_combos_that_im_in: Tuple[RoleAllocationEnum, ...] = tuple(
            [t for t in RoleAllocationEnum.__members__.values() if index in t.extract_sublist_from([0,1,2,3,4])]
        )
        """
        The role combos that include this agent.
        """
        self.the_other_role_combos: Tuple[RoleAllocationEnum, ...] = tuple(
            [t for t in RoleAllocationEnum.__members__.values() if t not in self.role_combos_that_im_in]
        )
        """
        The role combos that don't include this agent.
        """

        self.missions_that_i_sabotaged: List[int] = []
        """
        If I'm a spy, this is the list of missions that I sabotaged.
        """

        # noinspection PyTypeChecker
        self.other_spy: TPlayer = None
        """
        If I'm a spy, this is who the other spy is. If there is no other spy (because we're not a spy), this is None.
        """


    def onGameRevealed(self, players: List[TPlayer], spies: List[TPlayer]) -> None:
        """This function will be called to list all the players, and if you're
        a spy, the spies too -- including others and yourself.
        :param players:  List of all players in the game including you.
        :param spies:    List of players that are spies (if you are a spy), or an empty list (if you aren't a spy).
        """

        self.temp_team_record = TempTeamRecord()

        self.game_record.reset()

        self.missions_that_i_sabotaged.clear()

        self.player_records.clear()
        for p in players:
            self.player_records[p] = PlayerRecord(p)

        if self.spy:
            self.spies = spies
            # take note of the spies if we are a spy
            for p in players:
                is_spy: bool = p in spies
                self.player_records[p].identity_is_known(is_spy)
                if is_spy and p != self:
                    # we take note of who the other spy is.
                    self.other_spy = p

        self.role_combos_that_im_in = tuple(
            [t for t in RoleAllocationEnum.__members__.values() if self in t.extract_sublist_from(players)]
        )

        self.the_other_role_combos = tuple(
            [t for t in RoleAllocationEnum.__members__.values() if t not in self.role_combos_that_im_in]
        )

        self.suspicion_for_each_role_combo = dict.fromkeys(RoleAllocationEnum.__members__.values(), 1/6)
        for r in self.role_combos_that_im_in:
            self.suspicion_for_each_role_combo[r] = 0

        self.leader_order.clear()
        self.leader_order = players.copy()


        pass

    def onMissionAttempt(self, mission: int, tries: int, leader: TPlayer) -> None:
        """Callback function when a new turn begins, before the
        players are selected.
        :param mission:  Integer representing the mission number (1..5).
        :param tries:    Integer count for its number of tries (1..5).
        :param leader:   A Player representing who's in charge.
        """

        self.current_gamestate = GamestateTree.get_index_from_gamestate_object(self.game)  # index for current gamestate

        self.temp_team_record.reset_at_round_start(leader, mission, tries)

        # work out the order of leaders for this mission if this is the 1st attempt (start of a new mission)
        if tries == 1:
            self.leader_order.clear()
            for i in range(0, 4):
                self.leader_order.append(self.game.players[(leader.index + i) % len(self.game.players)])

        pass

    def select(self, players: List[TPlayer], count: int) -> List[TPlayer]:
        """
        Pick a sub-group of players to go on the next mission.

        Actually just a wrapper for self._select(players, count), but it shuffles the
        result of that, for the sole purpose of fucking around with any other bots that
        have logic that depends on the ordering of players within a team.

        :param players:  The list of all players in the game to pick from.
        :param count:    The number of players you must now select.
        :return: list    The players selected for the upcoming mission.
        """
        return random.sample(self._select(players, count), count)

    def _select(self, players: List[TPlayer], count: int) -> List[TPlayer]:
        """
        Pick a sub-group of players to go on the next mission.
        :param players:  The list of all players in the game to pick from.
        :param count:    The number of players you must now select.
        :return: list    The players selected for the upcoming mission.
        """

        team_list: List[TPlayer] = [self]

        if self.game.turn == 1:  # if turn 1, we pick the next leader to join us.
            team_list.append(players[(self.index + 1) % len(players)])
            return team_list

        elif count == 2:

            player_suspicions: Dict[TPlayer, float] = self.heuristic_suspicion_dict
            team_list.append(min(random.sample(self.others(), 4), key=lambda k: player_suspicions[k]))
            return team_list

        least_suspicious: List[TPlayer] = min(
            random.sample(self.the_other_role_combos, 6), key=lambda k: self.suspicion_for_each_role_combo[k]
        ).extract_sublist_from(players)

        #the_others: List[TPlayer] = self.others()
        return [self] + least_suspicious


    def onTeamSelected(self, leader: TPlayer, team: List[TPlayer]) -> None:
        """Called immediately after the team is selected to go on a mission,
        and before the voting happens.
        :param leader:   The leader in charge for this mission.
        :param team:     The team that was selected by the current leader.
        """
        self.temp_team_record.add_team_info(team, self.heuristic_suspicion_dict)
        pass

    def vote(self, team: List[TPlayer]) -> bool:
        """Given a selected team, decide whether the mission should proceed.
        :param team:      List of players with index and name.
        :return: bool     Answer Yes/No.
        """

        if self.game.tries == 5:
            if self.spy and self.game.turn == 5:
                return False  # if we're a spy and it's the final hammer, may as well vote no (nothing to lose)
            return True  # voting against a hammer is kinda sus ngl

        if self.spy:

            spies_on_team: int = len([s for s in self.spies if s in team])

            if spies_on_team == 0:
                # if there are no spies to sabotage this mission
                if self.game.wins == 2:
                    # always reject it if the resistance need one more pass to win
                    # as approving this team would lead to a loss.
                    return False

                # TODO: use a neural network to work out how likely it would be for a team to win the game
                #  if this team gets approved (non-sabotage from this gamestate) or rejected (from this gamestate).
                #  If a resistance win is more likely if approved, vote no. If a resistance win is more likely if
                #  rejected, vote yes.

                pass

            elif spies_on_team == 1:

                if self in team:
                    # if I am the spy in the team
                    if self.game.leader == self:
                        # if it's my team, I should vote for it.
                        return True
                    else:
                        # If I'm lucky enough to be on the team, I just act natural,
                        # blending in as a normal, rational, member of the resistance.
                        # If the team contains either of the two (other) players who are most suspicious now,
                        # I act like how any agent would in this situation, and vote against this team if it
                        # contains either of the two other players I currently suspect the most of being spies.
                        return self._vote_like_a_normal_resistance_member(team)

                else:  # if the other spy is on the team
                    if self.game.turn == 3 and self.game.losses == 2:
                        # may as well support it if it's turn 3 and we're one away from winning
                        return True
                    # otherwise, this spy would act like it normally would if it was a resistance member.
                    return self._vote_like_a_normal_resistance_member(team)


            else:
                # TODO: is it a good idea for me to vote for this team if there are two spies on it?

                # First things first: we act natural. If we have a decent reason to vote against it, vote against it.
                # there's a ~50% chance that we will have an excuse to vote against it.
                if not self._vote_like_a_normal_resistance_member(team):
                    return False

                if self.other_spy == self.game.leader:
                    # if the leader is the other spy, chances are that they have a cunning plan.
                    return True
                else:
                    # TODO use a neural network to work out if the optimal play here would be to
                    #  let the vote fail or to let the vote pass.
                    pass


                pass


            # TODO as spy, work out:
            #      is it safe to let the resistance win this round if there aren't any spies on the team?
            #      if this bot is on the team, is it safe to sabotage it without looking sus?
            #           and if there is another spy, should this agent sabotage, or let the other spy do it?
            return True


        # TODO as resistance, work out how likely the team is to have a spy who will sabotage it on it.

        # finds the most suspicious players (or, in other words, the most likely team of spies).
        # if either of them are on this team, we vote it down.
        return self._vote_like_a_normal_resistance_member(team)

    def _vote_like_a_normal_resistance_member(self, team: List[TPlayer]) -> bool:
        """
        Finds the most suspicious players/the most likely team of spies from the
        self.suspicion_for_each_role_combo dict.
        Then, it sees if either of those players are on the team.
        If either of them are on the team, well, that means one of the players we currently think is a spy
        is on the team. So we reject that team.
        :param team: the team we're voting on
        :return: true if that team doesn't contain either of the two players we currently think are
        most likely to be spies.
        """
        # finds the most suspicious players (or, in other words, the most likely team of spies).
        # if either of them are on this team, we vote it down.
        most_sus_players: List[TPlayer] = max(
            random.sample(self.the_other_role_combos, 6), key=lambda k: self.suspicion_for_each_role_combo[k]
        ).extract_sublist_from(self.game.players)

        if most_sus_players[0] in team or most_sus_players[1] in team:
            return False
        else:
            return True

    def onVoteComplete(self, votes: List[bool]) -> None:
        """Callback once the whole team has voted.
        :param: votes        Boolean votes for each player (ordered).
        """

        self.temp_team_record.add_vote_info(
            [p for p in self.game.players if votes[p.index]],
            sum(votes) >= 3
        )

        pass

    def sabotage(self) -> bool:
        """Decide what to do on the mission once it has been approved.  This
        function is only called if you're a spy, otherwise you have no choice.

        This is mostly just a wrapper for the _sabotage_logic method,
        which gets the result of that, adds the current gamestate id to
        the list of missions that this bot sabotaged (if true),
        and then returns that result.

        :return: bool        Yes to shoot down a mission.
        """
        sabotaging: bool = self._sabotage_logic()
        if sabotaging:
            self.missions_that_i_sabotaged.append(self.current_gamestate)
        return sabotaging

    def _sabotage_logic(self) -> bool:
        """
        The actual logic for the sabotage function.
        It's held in here, so the sabotage method can then easily get the result of this,
        and add it to the list of missions this bot sabotaged (if appropriate)
        :return: true to sabotage a mission.
        """
        spies_in_team: List[TPlayer] = [p for p in self.game.team if p in self.spies]
        if len(spies_in_team) > 1:
            if self.game.losses == 2 or self.game.wins == 2:
                # sabotage anyway if 2 rounds have already been sabotaged,
                # or if the resistance are guaranteed to win if nobody sabotages
                return True

            if self.game.leader == self.other_spy:
                # if the other spy is the leader, we let them sabotage instead.
                return False

            # works out how likely the other spy is to sabotage,
            return random.random() > rl18730._sabotage_chance_stats.get_sabotage_chance(
                self.other_spy.name,
                self.current_gamestate,
                self.game.leader == self.other_spy,
                True,
                self.game.leader in self.spies
            )

        # TODO: is it better to sabotage or to let this one through unsabotaged? might need to refer to the results
        #  from the NN earlier on.
        return True

    def onMissionComplete(self, sabotaged: int) -> None:
        """Callback once the players have been chosen.
        :param sabotaged:    Integer how many times the mission was sabotaged.
        """
        self.temp_team_record.add_mission_outcome_info(sabotaged)
        self._post_mission_housekeeping()
        pass

    def onMissionFailed(self, leader: TPlayer, team: List[TPlayer]) -> None:
        """Callback once a vote did not reach majority, failing the mission.
        :param leader:       The player responsible for selection.
        :param team:         The list of players chosen for the mission.
        """

        self._post_mission_housekeeping()
        pass

    def _post_mission_housekeeping(self) -> None:
        """
        This is where the bot performs any necessary housekeeping that needed to be done after the end
        of a mission
        (jobs such as exporting this round's team record, putting that on the history of team records,
        updating the player records with the results of this round, etc)

        :return:
        """

        # index_sabotage_tuple: Tuple[int, int] = (self.current_gamestate, this_round_record.sabotages)

        # TODO: replace the this_round_record stuff with data taken from the temp record (for now)
        for p in self.game.players:
            self.player_records[p].post_round_update(
                self.current_gamestate,
                self.temp_team_record.sabotages,
                len(self.game.team),
                p == self.game.leader,
                p in self.game.team,
                p in self.temp_team_record.voted_for_team
            )

        heuristic_suspicions: Dict[TPlayer, float] = self.heuristic_suspicion_dict

        #print("heuristic suspicions done.")

        self.temp_team_record.add_current_spy_probabilities(heuristic_suspicions)

        if not self.spy:
            heuristic_suspicions[self] = 0  # Why would I suspect myself of being a spy???
            # we only zero this if we're in the resistance, because spies need to worry more about not appearing too sus

        the_players: List[TPlayer] = self.game.players

        for rp in self.suspicion_for_each_role_combo.keys():
            sus_level: float = 1
            rp_list: Tuple[bool, bool, bool, bool, bool] = rp.get_value()
            for r in range(len(rp_list)):
                if rp_list[r]:
                    sus_level *= heuristic_suspicions[the_players[r]]
                else:
                    sus_level *= (1 - heuristic_suspicions[the_players[r]])
            self.suspicion_for_each_role_combo[rp] = sus_level

        # no need to bother normalizing the heuristic suspicions for our internal use



        #self.suspicion_for_each_role_combo[rp] = \
        #    sum(heuristic_suspicions[p] for p in those_players)/2
        # (heuristic_suspicions[those_players[0]] + heuristic_suspicions[the_players[rp[1]]]) / 2

        #print("CHANCES OF EACH TEAM BEING SPIES: ")
        #for rp in [*self.suspicion_for_each_role_combo.keys()]:
        #    print("{}: {}".format(rp, self.suspicion_for_each_role_combo[rp]))


        #this_round_record: TeamRecord = self.temp_team_record.generate_teamrecord_from_data

        # TODO: team record in the form
        #       * leader sus level
        #       * team sus levels
        #       * player sus levels
        #       * outcome (sabotages)
        #      so I can put that into some per-gamestate neural networks(?)

        self.game_record.add_teamrecord_to_records(
            self.current_gamestate,
            self.temp_team_record.generate_teamrecord_from_data
        )


        pass

    def announce(self) -> Dict[TPlayer, float]:
        """Publicly state beliefs about the game's state by announcing spy
        probabilities for any combination of players in the game.  This is
        done after each mission completes, and takes the form of a mapping from
        player to float.  Not all players must be specified, and of course this
        can be innacurate!

        :return: Dict[TPlayer, float]     Mapping of player to spy probability.
        """
        return {}

    def onAnnouncement(self, source: TPlayer, announcement: Dict[TPlayer, float]) -> None:
        """Callback if another player decides to announce beliefs about the
        game.  This is passed as a potentially incomplete mapping from player
        to spy probability.

        :param source:        Player making the announcement.
        :param announcement:  Dictionary mapping players to spy probabilities.
        """
        pass

    def say(self, message: str) -> None:
        """Helper function to print a message in the global game chat, visible
        by all the other players.

        :param message:       String containing free-form text.
        """
        self.log.info(message)

    def onMessage(self, source: TPlayer, message: str) -> None:
        """Callback if another player sends a general free-form message to the
        channel.  This is passed in as a generic string that needs to be parsed.

        :param source:       Player sending the message.
        :param message:  Arbitrary string for the message sent.
        """
        pass

    def onGameComplete(self, win: bool, spies: List[TPlayer]) -> None:
        """Callback once the game is complete, and everything is revealed.

        This is where stuff gets logged to rl18730.log.

        It's logged in the form:

        "teams"
            a dictionary of
                int keys -> gamestate id
                dict values -> teamrecord.loggable_dict outputs

        "res_win"
            boolean: whether or not the resistance won this game

        "spies"
            (bool, bool, bool, bool, bool)
                the values in this tuple for the players who are spies are set to true, false if resistance member.

        :param win:          Boolean true if the Resistance won.
        :param spies:        List of only the spies in the game.
        """

        known: List[int] = []

        for p in self.game.players:
            was_spy: bool = p in spies
            self.player_records[p].identity_is_known(was_spy)
            known.append(1 if was_spy else 0)
            # print(self.player_records[p].simple_spy_probability())
            # print(p in spies)

        log_dict: Dict[
            str,
            Union[
                Dict[
                    int,
                    Union[
                        Tuple[int, int, int, int, int],
                        Dict[str, float],
                        Tuple[float, float, float, float, float],
                        int
                    ]
                ],
                bool,
                Tuple[int, int, int, int, int]
            ]
        ] = {
            "teams": self.game_record.json_loggable_teamrecords,
            "res_win": win,
            "spies": tuple(known)
        }


        #for k in [*self.team_records.keys()]:
        #    log_dict["teams"][k] = self.team_records[k].loggable_dict

        self.log.debug(json.dumps(log_dict))

        rl18730._sabotage_chance_stats.add_sabotage_info(
            self.game_record.get_info_about_sabotages_from_spies_for_sabotage_records(
                [*spies], self, self.spy, self.missions_that_i_sabotaged
            )
        )

        """
        self._sabotage_chance_stats.add_sabotage_info(
            self.game_record.get_info_about_sabotages_from_spies_for_sabotage_records(
                [*spies], self, self.spy, self.missions_that_i_sabotaged
            )
        )
        """

        with open(resources_file_path/"sabotages.p", "wb") as p:
            pickle.dump(rl18730._sabotage_chance_stats, p)
            p.close()


        # for k in [*self.team_records.keys()]:
        #    print("{:3d} {}".format(k, self.team_records[k]))
        # print(json.dumps(log_dict))

        pass

    @property
    def heuristic_suspicion_dict(self) -> Dict[TPlayer, float]:
        """
        Obtains a dictionary with the basic heuristic suspicions for each player
        :return: dictionary of player-spy likelihood pairings
        """
        hsd: Dict[TPlayer, float] = {}
        for p in self.game.players:
            hsd[p] = self.player_records[p].simple_spy_probability()
        return hsd


