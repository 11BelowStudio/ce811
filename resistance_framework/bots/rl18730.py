from player import Bot, Player
from game import State

from typing import TypeVar, List, Dict, Set, Tuple, Iterable, FrozenSet, Union

from enum import Enum

import json

TPlayer = TypeVar("TPlayer", bound="Player")
"""A generic type for anything that might be a Player object"""

T = TypeVar("T")
"""A generic type that could be anything."""

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
        first_index: int = self.value.index(True)
        return "{}{}".format(first_index, self.value[first_index+1:].index(True) + first_index + 1)

    def from_jsoned_string(self, jsoned_string: str) -> "RoleAllocationEnum":
        """
        from a string ab (a = index of first true, b is index of second true),
        as would have been returned by to_string_for_json, and then returns the
        RoleAllocationEnum with the value that the string ab describes.
        Throws an exception if an unexpected input is given.
        :param jsoned_string: the sort of string that to_string_for_json would expect to be given
        :return: the RoleAllocationEnum which that string describes.
        """
        if len(jsoned_string) != 2:
            raise ValueError("You've given an invalid string, expected something like '01', got {}"
                             .format(jsoned_string)
                             )
        default_arr: List[bool, bool, bool, bool, bool] = [False, False, False, False, False]
        default_arr[int(jsoned_string[0])] = True
        default_arr[int(jsoned_string[1])] = True
        return RoleAllocationEnum(tuple(default_arr))


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
    def public_belief_states(self) -> Dict[RoleAllocationEnum, float]:
        """Public belief states(?) for this gamestate."""

        player_kv: List[TPlayer, float] = [*self._prior_predicted_spy_probabilities.items()]

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

        total_chances: float = sum([*public_state_dict.values()])

        if total_chances == 0:
            total_chances = 1

        for k in [*public_state_dict.keys()]:
            public_state_dict[k] /= total_chances  # sum of hypotheses = 1 (hopefully)

        return public_state_dict

    @property
    def json_dumpable_public_belief_states(self) -> Dict[str, float]:
        """
        Wrapper for self.public_belief_states that returns it in a format that's more json-friendly.
        Why?
        Attempting a json.dumps on the public_belief_states causes a
        'TypeError: keys must be str, int, float, bool or None, not RoleAllocationEnum' error message.
        So I'm converting the keys to str instead.
        :return: dict with keys ["ab"] where a = index of spy 1, b = index of spy 2.
        """
        pbs: Dict[RoleAllocationEnum, float] = self.public_belief_states

        jpbs: Dict[str, float] = {}

        for kv in [*pbs.items()]:
            jpbs[kv[0].to_string_for_json()] = kv[1]

        return jpbs



    def __str__(self):
        """Formats this as a string, shamelessly lifted from the game.State class"""
        output: str = "<TeamRecord\n"
        for key in sorted(self.__dict__):
            value = self.__dict__[key]
            output += "\t- %s: %r\n" % (key, value)
        output += "\t- %s: %r\n" % ("loggable_dict", self.loggable_dict)
        pbs: Dict[RoleAllocationEnum, float] = self.public_belief_states
        output += "\t- %s: %r\n" % ("public_belief_dict", pbs)
        output += "\t- %s: %r\n" % ("most_sus_pair", max([*pbs.items()], key=lambda kv: kv[1]))
        output += "\t- %s: %r\n" % ("public_belief_dict_json", self.json_dumpable_public_belief_states)
        return output + ">"

    @property
    def loggable_dict(self) -> Dict[str, Union[float, Tuple[int, float], Dict[int, float], int]]:
        """
        Attempts to turn this into a dict that can be logged
        :return: a dictionary with the following values:
        * p0
            * prior suspicion for player 0
        * p1
            * prior suspicion for player 1
        * p2
            * prior suspicion for player 2
        * p3
            * prior suspicion for player 3
        * p4
            * prior suspicion for player 4
        * leader
            * tuple of (leader index, leader suspicion)
            * a tuple of 4 0s and one 1 (with the 1 being in leader[self.leader.index])
        * beliefs
            * a dict of
                * RoleAllocationEnum.to_string_for_json()
                * float chance of each RoleAllocationEnum describing which team are spies
                    * normalized so the sum of all chances = 1
        * team
            * dictionary of {team member index: team member suspicion}
        * sabotaged
            * how many times the mission was sabotaged (0 if success, -1 if nomination failed)
        """
        info_dict: Dict[str, Union[float, Tuple[int, float], Dict[int, float], int]] = {}

        prior_probs: List[float] = [*self._prior_predicted_spy_probabilities.values()]

        for i in range(0, len(prior_probs)):
            info_dict["p{}".format(i)] = prior_probs[i]

        default_leader_array = [0, 0, 0, 0, 0]
        default_leader_array[self.leader.index] = 1

        # TODO: uncomment below line.
        #  info_dict["leader"] = tuple(default_leader_array)

        info_dict["leader"] = (self.leader.index, self._prior_predicted_spy_probabilities[self.leader])

        # TODO: uncomment the below line
        #  info_dict["beliefs"] = self.json_dumpable_public_belief_states

        info_dict["team"]: Dict[int, float] = {}

        for p in self.team:
            info_dict["team"][p.index] = info_dict["p{}".format(p.index)]
        #{p: self._prior_predicted_spy_probabilities[p] for p in self.team}

        info_dict["sabotaged"] = self._sabotages

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
        return [k for k in [*cls._node_dict.keys()] if cls._node_dict[k].hammer]

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
        """finds all non-rejected teams lead, and returns a list of [id for that gamestate, sabotages/participants]"""
        teams_sus: List[Tuple[int, float]] = []
        for t in self._missions_lead:
            sabs = self._all_missions_and_sabotages_with_teams[t]
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
    def mission_teams_been_on_sus_levels(self) -> List[Tuple[int, float]]:
        """finds all non-rejected teams been on, and returns a list of [id for that gamestate, sabotages/participants]"""
        teams_sus: List[Tuple[int, float]] = []
        for t in self._teams_been_on:
            sabs = self._all_missions_and_sabotages_with_teams[t]
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
            sabs = self._all_missions_and_sabotages_with_teams[t]
            if sabs[0] == -1:
                continue
            elif sabs[0] == 0:
                teams_sus.append((t, sabs[0]))
            else:
                teams_sus.append((t, sabs[0] / sabs[1]))
        return teams_sus

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
            return 0.25  # 0.5 suspicion if no data

        if len([h for h in self.hammers_thrown if h in all_prior_rounds]) > 0:
            return 1  # only a spy would throw a hammer vote.

        lead_sus: List[float] = [m[1] for m in self.mission_teams_lead_sus_levels if m[0] in all_prior_rounds]
        been_on_sus: List[float] = [m[1] for m in self.mission_teams_been_on_sus_levels if m[0] in all_prior_rounds]

        has_lead_missions: bool = len(lead_sus) > 0
        has_been_on_missions: bool = len(been_on_sus) > 0

        # voted_for_sus: List[float] = [m[1] for m in self.approved_teams_sus_levels if m[0] in all_prior_rounds]
        # voted_against_sus: List[float] = [m[1] for m in self.rejected_teams_sus_levels if m[0] in all_prior_rounds]

        # total_concluded_missions_voted_on: int = len(voted_for_sus) + len(voted_against_sus)

        if has_lead_missions or has_been_on_missions:
            lead_len = len(lead_sus)
            been_len = len(been_on_sus)
            return min((sum(lead_sus)/4) + (sum(been_on_sus)) / (lead_len/4 + been_len), 1)
            #  return (sum(lead_sus) * max(1,lead_len)) + (sum(been_on_sus) * max(1, been_len)) / (lead_len + been_len) * (max(1, lead_len) * max(1, been_len))
        else:
            return 0.25
        #elif total_concluded_missions_voted_on > 0:
        #    for_against_sus: float = max(
        #        (
        #                (sum(voted_for_sus) * len(voted_for_sus)) - (sum(voted_against_sus) * len(voted_against_sus))
        #        ) / total_concluded_missions_voted_on, 0)
        #    return for_against_sus #  0.25




class rl18730(Bot):
    """
    Rachel's somewhat functional bot for playing The Resistance.

    note: currently not functional.
    """


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

    def onGameRevealed(self, players: List[TPlayer], spies: List[TPlayer]) -> None:
        """This function will be called to list all the players, and if you're
        a spy, the spies too -- including others and yourself.
        :param players:  List of all players in the game including you.
        :param spies:    List of players that are spies (if you are a spy), or an empty list (if you aren't a spy).
        """

        self.temp_team_record = TempTeamRecord()

        self.player_records.clear()
        for p in players:
            self.player_records[p] = PlayerRecord(p)

        if self.spy:
            self.spies = spies
            # take note of the spies if we are a spy
            for p in players:
                self.player_records[p].identity_is_known(p in spies)

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
        """Pick a sub-group of players to go on the next mission.
        :param players:  The list of all players in the game to pick from.
        :param count:    The number of players you must now select.
        :return: list    The players selected for the upcoming mission.
        """

        team_list: List[TPlayer] = [self]

        if self.game.turn == 1:  # if turn 1, we pick the next leader to join us.
            team_list.append(self.game.players[(self.index + 1) % len(self.game.players)])
            return team_list

        elif count == 2:

            player_suspicions: Dict[TPlayer, float] = self.heuristic_suspicion_dict
            team_list.append(min(self.others(), key=lambda k: player_suspicions[k]))
            return team_list

        least_suspicious: List[TPlayer] = min(
            self.the_other_role_combos, key=lambda k: self.suspicion_for_each_role_combo[k]
        ).extract_sublist_from(self.game.players)

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
            if not self.spy:
                return True
            if self.game.turn == 5:
                return False  # if we're a spy and it's the final hammer, may as well vote no.

        if self.spy:
            # TODO as spy, work out:
            #      is it safe to let the resistance win this round if there aren't any spies on the team?
            #      if this bot is on the team, is it safe to sabotage it without looking sus?
            #           and if there is another spy, should this agent sabotage, or let the other spy do it?
            return True


        # TODO as resistance, work out how likely the team is to have a spy who will sabotage it on it.

        # finds the most suspicious players (or, in other words, the most likely team of spies).
        # if either of them are on this team, we vote it down.
        most_sus_players: List[TPlayer] = max(
            self.the_other_role_combos,key=lambda k: self.suspicion_for_each_role_combo[k]
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
        :return: bool        Yes to shoot down a mission.
        """
        spies_in_team: List[TPlayer] = [p for p in self.game.team if p in self.spies]
        if len(spies_in_team) > 1:
            # TODO: work out probability of other spies in team sabotaging
            pass
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

        heuristic_suspicions[self] = 0  # Why would I suspect myself of being a spy???

        the_players: List[TPlayer] = self.game.players



        for rp in [*self.suspicion_for_each_role_combo.keys()]:
            # those_players: List[TPlayer] = rp.extract_sublist_from(the_players)
            sus_level: float = 1
            rp_list: Tuple[bool, bool, bool, bool, bool] = rp.get_value()
            if rp_list in self.role_combos_that_im_in:
                self.suspicion_for_each_role_combo[rp] = 0  # I'm not a spy, why would I suspect myself???
                continue
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


        this_round_record: TeamRecord = self.temp_team_record.generate_teamrecord_from_data

        # TODO: team record in the form
        #       * leader sus level
        #       * team sus levels
        #       * player sus levels
        #       * outcome (sabotages)
        #      so I can put that into some per-gamestate neural networks(?)

        self.team_records[self.current_gamestate] = this_round_record


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

        known: List[bool] = []

        for p in self.game.players:
            was_spy: bool = p in spies
            self.player_records[p].identity_is_known(was_spy)
            known.append(was_spy)
            #print(self.player_records[p].simple_spy_probability())
            #print(p in spies)

        for k in [*self.team_records.keys()]:
            print("{:3d} {}".format(k, self.team_records[k]))

        log_dict: Dict[str, Union[
            Dict[int, Dict[str, Union[float, Tuple[int, float], Dict[int, float], int]]],
            bool,
            Tuple[bool, bool, bool, bool, bool]
        ]] = {
            "teams": {},
            "res_win": win,
            "spies": tuple(known)
        }

        for k in [*self.team_records.keys()]:
            log_dict["teams"][k] = self.team_records[k].loggable_dict

        self.log.debug(json.dumps(log_dict))

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


