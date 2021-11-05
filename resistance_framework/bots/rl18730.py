
from player import Bot, Player
from game import State

from typing import TypeVar, List, Dict, Set, Tuple, Iterable, FrozenSet

TPlayer = TypeVar("TPlayer", bound="Player")


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

        self._team: FrozenSet[TPlayer] = frozenset(team)
        self._leader: TPlayer = leader
        self._mission_number: int = mission_number
        self._nomination_attempt: int = nomination_attempt
        self._nomination_successful: bool = nomination_successful
        self._sabotages: int = sabotages if nomination_successful else -1
        self._voted_for_team: FrozenSet[TPlayer] = frozenset(voted_for_team)

    @property
    def team(self) -> FrozenSet[TPlayer]:
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
        """How many times was the mission sabotaged? (-1 if it didn't happen)"""
        return self._sabotages

    @property
    def voted_for_team(self) -> FrozenSet[TPlayer]:
        """Who voted in favour of the team?"""
        return self._voted_for_team



class GamestateTree(object):
    """
    In short, this is a data structure which exists to store gamestate info as an int,
    index a tree of possible gamestates,
    and also

    win offset + 3
    loss: offset - 1

                -3
            -2      -3 (0)
        -1      1       -3 (3)
    0       2       4
        3       5       9  (7)
            6       9   (8)
                9


                -15
            -10     -15 (0)
        -5      5       -15 (15)
    0       10       20
        15      25       35
            30       35 (40)
                35 (45)

    here's a crappy visual representation of how I've indexed the gamestates.
    up: losing. down: winning.


    after a loss, next round's attempt indexes follow on from 'currentRoundFinalAttempt + 1'
    after a win, next round's attempt indexes start from 'currentRoundFinalAttempt + 16'
    15, 35, 55 are all 'spy victory'.
    60, 65, 70 are all 'resistance victory'.

    Resistance wants to get to the highest possible index (greedily).
    Spies want to get to the smallest possible index.

    easiest way of assigning keys to them in a way that might make a little bit of sense.
    m1  m2  m3  m4  m5
                15 (spy win)
            10-14   35 (spy win)
        5-9     30-34   55 (spy win)
    0-4     25-29   50-54
        20-24   45-49   70 (resistance win)
            40-44   65 (resistance win)
                60 (resistance win)
    """

    class gamestate_tree_node(object):
        """
        Attempts to

        Recalls what index within the tree this index has,
        along with int pointers to the indexes that hold the nodes
        which must be navigated to by the tree traversal algorithm
        when either this proposal is rejected, or when the mission associated with this
        proposal passes or fails.

        Might try to incorporate an NN into this which analyses, given the gamestate,
        what is likely to happen at this point (proposal fail/mission pass/mission fail)
        given the history up to this point, and what the best way to vote for the mission would be.
        """

        def __init__(self, index: int, voteFailedChild: int, missionPassedChild: int, missionFailedChild: int):
            self._index = index
            self._voteFailedChild: int = voteFailedChild
            self._missionPassedChild: int = missionPassedChild
            self._missionFailedChild: int = missionFailedChild
            self._traversals: int = 0
            self._encountered: int = 0

        def __repr__(self):
            return "Index: {:2d}, Reject {:2d}, Pass {:2d}, Fail {:2d}" \
                .format(self._index, self._voteFailedChild, self._missionPassedChild, self._missionFailedChild)

        @property
        def index(self) -> int:
            """index of this node"""
            return self._index

    _spy_win_offset: int = -5
    """Offset for spy wins"""
    _res_win_offset: int = 15
    """Offset for resistance wins"""

    _spy_win_state_indices: Tuple[int, int, int] = (15, 35, 55)
    """Indices of spy win states"""
    _res_win_state_indices: Tuple[int, int, int] = (60, 65, 70)
    """Indices of resistance win states"""

    _node_dict: Dict[int, "GamestateTree.gamestate_tree_node"] = {}
    for i in range(-3, 7):
        # 11 'groups' of actual gamestates that matter (0-55 are nomination states (except 15-35)

        # these groups are reserved for spy win conditions.
        if i == -1:
            continue

        nom1: int = i * 5
        rangeEnd: int = nom1 + 5
        step: int = 1

        if i < 0:
            step = rangeEnd
            rangeEnd = nom1
            nom1 = step
            step = -1

        nomination1: int = i * 5
        spy_win: int = nomination1 + _spy_win_offset
        for g in range(nom1, rangeEnd, step):
            # creates a gamestate index node for the current proposal.
            # located at index g.
            # refusing proposal redirects to the node at g+1
            # failing the mission redirects to the node at i15 (5 ahead of the first index of this group)
            # passing the mission redirects to the node at 15 + 20 (20 ahead of the first index of this group)

            # TODO: actually it might make more sense to have all the individual failed nominations go to a -1,
            #       because failed vote means potential loss of utility to resistance, closer to spy win, etc?
            spyWinIndex = nom1 - 5
            if spyWinIndex == 0 or spyWinIndex == 15:
                spyWinIndex = -15
            resWinIndex = nom1 + 15
            if resWinIndex > 30:
                resWinIndex = 35

            failIndex = g + step
            if failIndex == rangeEnd:
                failIndex = spyWinIndex

            _node_dict[g] = gamestate_tree_node(g, failIndex, resWinIndex, spyWinIndex)


    def to_gamestate_index(self, res_wins: int, spy_wins: int, nomination_attempt: int) -> int:
        what_round: int = (res_wins * GamestateTree._res_win_offset) + (spy_wins * GamestateTree._spy_win_offset)

        if what_round < 0:
            return what_round - nomination_attempt
        else:
            return what_round + nomination_attempt


for k in [*GamestateTree._node_dict.keys()]:
    print("{:3d}: {}".format(k, GamestateTree._node_dict[k]))







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

        self.spies: Set[TPlayer] = set()
        """Set of known spies (empty unless spy)"""

        self.team_records: Dict[int, TeamRecord]



    def onGameRevealed(self, players: List[TPlayer], spies: List[TPlayer]) -> None:
        """This function will be called to list all the players, and if you're
        a spy, the spies too -- including others and yourself.
        :param players:  List of all players in the game including you.
        :param spies:    List of players that are spies (if you are a spy), or an empty list (if you aren't a spy).
        """

        pass

    def onMissionAttempt(self, mission: int, tries: int, leader: TPlayer) -> None:
        """Callback function when a new turn begins, before the
        players are selected.
        :param mission:  Integer representing the mission number (1..5).
        :param tries:    Integer count for its number of tries (1..5).
        :param leader:   A Player representing who's in charge.
        """
        pass

    def select(self, players: List[TPlayer], count: int) -> List[TPlayer]:
        """Pick a sub-group of players to go on the next mission.
        :param players:  The list of all players in the game to pick from.
        :param count:    The number of players you must now select.
        :return: list    The players selected for the upcoming mission.
        """
        raise players[:count]

    def onTeamSelected(self, leader: TPlayer, team: List[TPlayer]) -> None:
        """Called immediately after the team is selected to go on a mission,
        and before the voting happens.
        :param leader:   The leader in charge for this mission.
        :param team:     The team that was selected by the current leader.
        """
        pass

    def vote(self, team: List[TPlayer]) -> bool:
        """Given a selected team, decide whether the mission should proceed.
        :param team:      List of players with index and name.
        :return: bool     Answer Yes/No.
        """
        raise True

    def onVoteComplete(self, votes: List[bool]) -> None:
        """Callback once the whole team has voted.
        :param: votes        Boolean votes for each player (ordered).
        """
        pass

    def sabotage(self) -> bool:
        """Decide what to do on the mission once it has been approved.  This
        function is only called if you're a spy, otherwise you have no choice.
        :return: bool        Yes to shoot down a mission.
        """
        raise True

    def onMissionComplete(self, sabotaged: int) -> None:
        """Callback once the players have been chosen.
        :param sabotaged:    Integer how many times the mission was sabotaged.
        """
        pass

    def onMissionFailed(self, leader: TPlayer, team: List[TPlayer]) -> None:
        """Callback once a vote did not reach majority, failing the mission.
        :param leader:       The player responsible for selection.
        :param team:         The list of players chosen for the mission.
        """
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
        pass



