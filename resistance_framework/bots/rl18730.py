from player import Bot, Player
from game import State

from typing import TypeVar, List, Dict, Set, Tuple, Iterable, FrozenSet

TPlayer = TypeVar("TPlayer", bound="Player")

# Hi, my name is Rachel, and welcome to Jackass.
# at least there are type annotations :^)


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
                 voted_for_team: Iterable[TPlayer]
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
        """

        self._team: Tuple[TPlayer, ...] = tuple(team)
        self._leader: TPlayer = leader
        self._mission_number: int = mission_number
        self._nomination_attempt: int = nomination_attempt
        self._nomination_successful: bool = nomination_successful
        self._sabotages: int = sabotages if nomination_successful else -1
        self._voted_for_team: FrozenSet[TPlayer] = frozenset(voted_for_team)

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

    def __str__(self):
        """Formats this as a string, shamelessly lifted from the game.State class"""
        output: str = "<TeamRecord\n"
        for key in sorted(self.__dict__):
            value = self.__dict__[key]
            output += "\t- %s: %r\n" % (key, value)
        return output + ">"


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
        self.sabotages: int = -1
        self.voted_for_team: FrozenSet[TPlayer] = frozenset()

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
        self.sabotages = -1
        self.voted_for_team = ()

    def add_team_info(self, team: Iterable[TPlayer]) -> None:
        """
        Copy team info to this object when the team is revealed
        :param team: the new team
        """
        self.team = tuple(team)

    def add_vote_info(self, yes_men: Iterable[TPlayer], approved: bool) -> None:
        """
        Copies info about the vote to the team record
        :param yes_men: players who voted in favour of it
        :param approved: true if the vote passed
        """
        self.voted_for_team = frozenset(yes_men)
        self.nomination_successful = approved

    def add_mission_outcome_info(self, sab: int) -> None:
        """
        Adds the mission outcome info to the TeamRecord
        :param sab: number of times this mission was sabotaged
        """
        self.sabotages = sab

    def generate_teamrecord_from_data(self) -> TeamRecord:
        """
        Puts the data in this TempTeamRecord into a proper TeamRecord so it can be saved for later
        :return: a TeamRecord with the current data from this TempTeamRecord
        """
        return TeamRecord(self.team, self.leader, self.mission_number, self.nomination_attempt,
                          self.nomination_successful, self.sabotages, self.voted_for_team)


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

        def __str__(self):
            return "Index: {:3d}, Reject {:3d}, Pass {:3d}, Fail {:3d}" \
                .format(self._index, self._voteFailedChild, self._missionPassedChild, self._missionFailedChild)

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
    def get_hammer_indices(cls) -> List[int]:
        """
        Indices for the final nomination attempts
        """
        return [k for k in [*cls._node_dict.keys()] if cls._node_dict[k].hammer]

    @classmethod
    def get_team_size_from_index(cls, ind: int) -> int:
        """Returns the team size for the gamestate at the given index.
        If given gamestate is not a known gamestate, this just returns 3 instead because I'm lazy."""

        indMinus1Mod5: int = (ind-1) % 5
        if indMinus1Mod5 > 4 and ind - indMinus1Mod5 != 11:
            # basically everything that's range(0, -5, -1) has a team size of 2, except range(20, 16, -1)
            return 2
        return 3

    @classmethod
    def get_raw_regret_from_index(cls, ind: int) -> Tuple[int, int, int]:
        """
        Works out 'raw' counterfactual regret for the actions that could follow from this gamestate.
        I'm referring it it as 'raw', because it's just the regret associated with each outcome
        (success/sabotaged/rejection) from the current gamestate, and something else will be
        calculating the actual probabilities of each outcome happening (which these values can simply be multiplied
        by later on)
        :param ind: index of the parent node that we're trying to work out the immediate counterfactual regret of
        :return: tuple with (reject regret, success regret, sabotage regret)
        """

        if ind == cls._spy_win_index or ind == cls._res_win_index:
            return 0, 0, 0  # not much left to regret if the game is already over.

        this_state_node: "GamestateTree.GamestateTreeNode" = cls._node_dict[ind]

        return (this_state_node.voteFailedChild - ind,
                this_state_node.missionPassedChild - ind,
                this_state_node.missionFailedChild - ind)


# a little bit of cleanup on the gamestatetree, removing a couple of unwanted static variables that hung around
# noinspection PyBroadException
try:
    delattr(GamestateTree, "nom1")
    delattr(GamestateTree, "rangeEnd")
    delattr(GamestateTree, "step")
except Exception:
    pass


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

        self._all_missions_and_sabotages: Dict[int, int] = {}
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

    def post_round_update(self, index: int, sab: int, was_leader: bool,
                          was_on_team: bool, voted_for_team: bool) -> None:
        """
        Call this to update the player info with the data for each round.
        :param index: mission ID (via GamestateTree indices)
        :param sab: sabotage count (-1 if team rejected)
        :param was_leader: true if this player was the leader of the team.
        :param was_on_team: true if this player was leading this team.
        :param voted_for_team: true if this player voted for the team.
        :return: nothing.
        """

        self._all_missions_and_sabotages[index] = sab

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
        return [m for m in self._missions_lead if self._all_missions_and_sabotages[m] == 0]

    @property
    def failed_missions_lead(self) -> List[int]:
        """All failed missions that this player lead"""
        return [m for m in self._missions_lead if self._all_missions_and_sabotages[m] > 0]

    @property
    def rejected_missions_lead(self) -> List[int]:
        """All rejected teams that this player nominated"""
        return [m for m in self._missions_lead if self._all_missions_and_sabotages[m] == -1]

    @property
    def teams_been_on(self) -> List[int]:
        """All teams that this player has been on"""
        return self._teams_been_on.copy()

    @property
    def passed_teams_been_on(self) -> List[int]:
        """All passed missions that this player was on"""
        return [m for m in self._teams_been_on if self._all_missions_and_sabotages[m] == 0]

    @property
    def failed_teams_been_on(self) -> List[int]:
        """All failed missions that this player was on"""
        return [m for m in self._teams_been_on if self._all_missions_and_sabotages[m] > 0]

    @property
    def rejected_teams_been_on(self) -> List[int]:
        """All rejected teams that this player was on"""
        return [m for m in self._teams_been_on if self._all_missions_and_sabotages[m] == -1]

    @property
    def teams_approved(self) -> List[int]:
        """All teams that this player voted for"""
        return self._teams_approved.copy()

    @property
    def passed_teams_approved(self) -> List[int]:
        """All passed missions that this player voted for"""
        return [m for m in self._teams_approved if self._all_missions_and_sabotages[m] == 0]

    @property
    def failed_teams_approved(self) -> List[int]:
        """All failed missions that this player voted for"""
        return [m for m in self._teams_approved if self._all_missions_and_sabotages[m] > 0]

    @property
    def rejected_teams_approved(self) -> List[int]:
        """All rejected teams that this player voted for"""
        return [m for m in self._teams_approved if self._all_missions_and_sabotages[m] == -1]

    @property
    def teams_rejected(self) -> List[int]:
        """All teams that this player voted against"""
        return self._teams_rejected.copy()

    @property
    def passed_teams_rejected(self) -> List[int]:
        """All passed missions that this player voted against"""
        return [m for m in self._teams_rejected if self._all_missions_and_sabotages[m] == 0]

    @property
    def failed_teams_rejected(self) -> List[int]:
        """All failed missions that this player voted against"""
        return [m for m in self._teams_rejected if self._all_missions_and_sabotages[m] > 0]

    @property
    def rejected_teams_rejected(self) -> List[int]:
        """All rejected teams that this player voted against"""
        return [m for m in self._teams_rejected if self._all_missions_and_sabotages[m] == -1]

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

    @property
    def spy_probability(self, current_round_index: int = -1) -> float:
        """
        Works out how likely this player is to be a spy, given their actions until the current round.
        :param current_round_index: Index of the current round (GamestateTree index). Defaults to -1.
        If negative/unspecified/not an index of a round that has been reached, takes everything
        known so far into account

        :return: a float indicating how suspicious this player currently is.
        """
        if len(self._all_missions_and_sabotages) == 0:
            return 0.5

        # TODO actual calculations of suspiciousness. Could try to use some neural networks/bayesian belief stuff/etc.



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

    def onGameRevealed(self, players: List[TPlayer], spies: List[TPlayer]) -> None:
        """This function will be called to list all the players, and if you're
        a spy, the spies too -- including others and yourself.
        :param players:  List of all players in the game including you.
        :param spies:    List of players that are spies (if you are a spy), or an empty list (if you aren't a spy).
        """
        for p in players:
            self.player_records[p] = PlayerRecord(p)

        if self.spy:
            self.spies = spies
            # take note of the spies if we are a spy
            for p in players:
                self.player_records[p].identity_is_known(p in spies)

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

        pass

    def select(self, players: List[TPlayer], count: int) -> List[TPlayer]:
        """Pick a sub-group of players to go on the next mission.
        :param players:  The list of all players in the game to pick from.
        :param count:    The number of players you must now select.
        :return: list    The players selected for the upcoming mission.
        """
        return players[:count]

    def onTeamSelected(self, leader: TPlayer, team: List[TPlayer]) -> None:
        """Called immediately after the team is selected to go on a mission,
        and before the voting happens.
        :param leader:   The leader in charge for this mission.
        :param team:     The team that was selected by the current leader.
        """
        self.temp_team_record.add_team_info(team)
        pass

    def vote(self, team: List[TPlayer]) -> bool:
        """Given a selected team, decide whether the mission should proceed.
        :param team:      List of players with index and name.
        :return: bool     Answer Yes/No.
        """
        if self.spy:
            # TODO as spy, work out:
            #      is it safe to let the resistance win this round if there aren't any spies on the team?
            #      if this bot is on the team, is it safe to sabotage it without looking sus?
            #           and if there is another spy, should this agent sabotage, or let the other spy do it?
            return True
        else:
            if self.game.tries == 5:
                return True
            # TODO as resistance, work out how likely the team is to have a spy who will sabotage it on it.

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
        return True

    def onMissionComplete(self, sabotaged: int) -> None:
        """Callback once the players have been chosen.
        :param sabotaged:    Integer how many times the mission was sabotaged.
        """
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
        this_round_record: TeamRecord = self.temp_team_record.generate_teamrecord_from_data()

        # index_sabotage_tuple: Tuple[int, int] = (self.current_gamestate, this_round_record.sabotages)

        for p in self.game.players:
            self.player_records[p].post_round_update(
                self.current_gamestate,
                this_round_record.sabotages,
                p == this_round_record.leader,
                p in this_round_record.team,
                p in this_round_record.voted_for_team
            )

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
        :param win:          Boolean true if the Resistance won.
        :param spies:        List of only the spies in the game.
        """

        for p in self.game.players:
            self.player_records[p].identity_is_known(p in spies)

        for k in [*self.team_records.keys()]:
            print("{:3d} {}".format(k, self.team_records[k]))

        pass
