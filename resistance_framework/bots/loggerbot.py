from player import Bot, TPlayer
from game import State
import random
from typing import List, Dict, Set

# run this with python competition.py 10000 bots/intermediates.py bots/loggerbot.py  
# Then check logs/loggerbot.log   Delete that file before running though

class LoggerBot(Bot):

    # Loggerbot makes very simple playing strategy.
    # We're not really trying to win here, but just to observer the other players
    # without disturbing them too much....

    def __init__(self, game: State, index: int, spy: bool):
        super(LoggerBot, self).__init__(game, index, spy)

        self.failed_missions_been_on: Dict[TPlayer, int] = {}
        """Dictionary that keeps count of how many times
        each player has been on a team that failed."""
        self.missions_been_on: Dict[TPlayer, int] = {}
        """Dictionary with count of how many times each player
        has been on a mission"""

        self.num_missions_voted_up_with_total_suspect_count: Dict[
            TPlayer, List[int, int, int, int, int, int]
        ] = {}
        """
        Dictionary that holds, for each player, a list of 6 ints,
        where the ith element shows how many teams with a suspect
        count of i that the player voted in favour of
        """

        self.num_missions_voted_down_with_total_suspect_count: Dict[
            TPlayer, List[int, int, int, int, int, int]
        ] = {}
        """
        Dictionary that holds, for each player, a list of 6 ints,
        where the ith element shows how many teams with a suspect
        count of i that the player voted against
        """

    def select(self, players: List[TPlayer], count: int) -> List[TPlayer]:
        return [self] + random.sample(self.others(), count - 1)

    def vote(self, team: List[TPlayer]) -> bool:
        return True

    def sabotage(self) -> bool:
        return True

    def mission_total_suspect_count(self, team: List[TPlayer]) -> int:
        """
        Returns the total number of failed missions that the players in the
        current team have been on
        :param team: the current team
        :return: total failed missions for everyone on the team
        """
        sus_count: int = 0
        for p in team:
            sus_count += self.failed_missions_been_on[p]
        return sus_count
        
    def onVoteComplete(self, votes: List[bool]) -> None:
        """Callback once the whole team has voted.
        @param votes        Boolean votes for each player (ordered).
        """
        pass # TODO complete this function

    def onGameRevealed(self, players: List[TPlayer], spies: List[TPlayer]) -> None:
        """This function will be called to list all the players, and if you're
        a spy, the spies too -- including others and yourself.
        :param players:  List of all players in the game including you.
        :param spies:    List of players that are spies (if you are a spy), or an empty list (if you aren't a spy).
        """
        self.failed_missions_been_on.clear()
        self.missions_been_on.clear()
        self.num_missions_voted_up_with_total_suspect_count.clear()
        self.num_missions_voted_down_with_total_suspect_count.clear()
        for p in players:
            self.failed_missions_been_on[p] = 0
            self.missions_been_on[p] = 0
            self.num_missions_voted_up_with_total_suspect_count[p] = [0, 0, 0, 0, 0, 0]
            self.num_missions_voted_down_with_total_suspect_count[p] = [0, 0, 0, 0, 0, 0]

        pass # TODO complete this function

    def onMissionComplete(self, sabotaged: int) -> None:
        """Callback once the players have been chosen.
        @param sabotaged    Integer how many times the mission was sabotaged.
        """
        pass # TODO complete this function

    def onGameComplete(self, win: bool, spies: List[TPlayer]) -> None:
        """Callback once the game is complete, and everything is revealed.
        @param win          Boolean true if the Resistance won.
        @param spies        List of only the spies in the game.
        """
        pass # TODO complete this function (in challenge 2)
