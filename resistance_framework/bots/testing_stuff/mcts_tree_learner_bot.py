from player import Bot, Player
from game import State

from typing import TypeVar, List, Dict, Set

TPlayer = TypeVar("TPlayer", bound="Player")



class mcts_tree(object):
    """
    win offset +1
    loss: offset + 4
                3
            2       7
        1       6       11
    0       5       10
        4       9       14
            8       13
                12

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

    def __init__(self):
        self.node_dict: Dict[int, "mcts_tree_node"] = {}
        #for i in range(0, 11):
        for i in range(-3, 7):

            if i == -1:
                continue

            # these groups are reserved for spy win conditions.
            #if i == 3 or i == 7:
            #    continue

            i5: int = i * 5
            i15: int = i5 + 5
            step: int = 1
            negativeAdjustment: int = 0

            if i < 0:
                step = i15
                i15 = i5
                i5 = step
                step = -1
                #negativeAdjustment = -4

            print(range(i5, i15, step))

            for g in range(i5, i15, step):
                # creates a mcts node for the current proposal.
                # located at index g.
                # refusing proposal redirects to the node at g+1
                # failing the mission redirects to the node at i15 (5 ahead of the first index of this group)
                # passing the mission redirects to the node at 15 + 20 (20 ahead of the first index of this group)
                spyWinIndex = i5 - 5
                if spyWinIndex == 0 or spyWinIndex == 15:
                    spyWinIndex = -15
                resWinIndex = i5 + 15
                if resWinIndex > 30:
                    resWinIndex = 35

                failIndex = g+step
                if failIndex == i15:
                    failIndex = spyWinIndex

                self.node_dict[g] = mcts_tree.mcts_tree_node(g, failIndex, resWinIndex, spyWinIndex)


        for k in [*self.node_dict.keys()]:
            print("{:3d}: {}".format(k, self.node_dict[k]))

    class mcts_tree_node(object):
        """
        A node in a monte carlo tree search tree.
        Represents an individual proposal gamestate within a game of The Resistance

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
            self.index = index
            self.voteFailedChild: int = voteFailedChild
            self.missionPassedChild: int = missionPassedChild
            self.missionFailedChild: int = missionFailedChild
            self.traversals: int = 0
            self.encountered: int = 0

        def __repr__(self):
            return "Index: {:3d}, Reject {:3d}, Pass {:3d}, Fail {:3d}" \
                .format(self.index, self.voteFailedChild, self.missionPassedChild, self.missionFailedChild)


mcts = mcts_tree()





class mcts_tree_learner_bot(Bot):
    """
    Attempts to learn stuff for neural networks that can be used in a MCTS tree



    loss: offset +1
    win : offset +4
                3
            2       7
        1       6       11
    0       5       10
        4       9       14
            8       13
                12

    here's a crappy visual representation of how I've indexed the gamestates.
    up: losing. down: winning.


    after a loss, next round's attempt indexes follow on from 'currentRoundFinalAttempt + 1'
    after a win, next round's attempt indexes start from 'currentRoundFinalAttempt + 16'
    15, 35, 55 are all 'spy victory'.
    60, 65, 70 are all 'resistance victory'.

    Resistance wants to get to the highest possible index.
    Spies want to get to the lowest possible index.

    easiest way of assigning keys to them in a way that might make a little bit of sense.
    m1  m2  m3  m4  m5
                15
            10-14   35
        5-9     30-34   55
    0-4     25-29   50-54
        20-24   45-49   70
            40-44   65
                60


    """

    def __init__(self, game: State, index: int, spy: bool):
        """Constructor called before a game starts.  It's recommended you don't
        override this function and instead use onGameRevealed() to perform
        setup for your AI.
        :param game:     the current game state
        :param index:    Bot's index in the player list.
        :param spy:      Is this bot meant to be a spy?
        """
        super().__init__(game, index, spy)



    def onGameRevealed(self, players: List[TPlayer], spies: Set[TPlayer]) -> None:
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

    def onGameComplete(self, win: bool, spies: Set[TPlayer]) -> None:
        """Callback once the game is complete, and everything is revealed.
        :param win:          Boolean true if the Resistance won.
        :param spies:        List of only the spies in the game.
        """
        pass


