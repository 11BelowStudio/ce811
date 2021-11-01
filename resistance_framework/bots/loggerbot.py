import sys

from player import Bot, TPlayer
from game import State
import random
from typing import List, Dict, Iterable, Set

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

        self.training_feature_vectors: Dict[TPlayer, List[
            List[
                int, int, int, str, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int
            ]
        ]] = {}
        """
        A dictionary that will hold feature vectors for each player,
        for use with a neural network.
        * turn
        * tries
        * index
        * name (string)
        * missions been on
        * failed missions been on
        * failed missions proposed
        * missions with each suspect count voted in favour of (6)
        * missions with each suspect count voted against (6)
        * whether or not it's a spy
        """
        self.spies: List[TPlayer] = []
        """Will hold all known spies (empty if not a spy)"""
        self.failed_missions_proposed: Dict[TPlayer, int] = {}
        """a dictionary keeping track of every failed mission proposed by the given player"""

    def select(self, players: List[TPlayer], count: int) -> List[TPlayer]:
        sus_dict: Dict[TPlayer, float] = self.get_sus_dict()
        sorted_players_by_trustworthiness: List[TPlayer] = \
            sorted(players, key=lambda p1: sus_dict[p1])
        sorted_players_by_trustworthiness.remove(self)

        return [self] + sorted_players_by_trustworthiness[:count - 1]
        #return [self] + random.sample(self.others(), count - 1)

    def vote(self, team: List[TPlayer]) -> bool:
        """
        sus_levels: Dict[TPlayer, float] = self.get_sus_dict()
        sorted_players_by_trustworthiness: List[TPlayer] = \
            sorted([*sus_levels.keys()], key=lambda p2: sus_levels[p2])
        if not self.spy:
            if self.game.tries == 5:
                return True
            if self in team: # if this bot is in the team, just look at the other players in the team
                theOthers: List[TPlayer] = team.copy()
                theOthers.remove(self)
                sorted_players_by_trustworthiness.remove(self)
                for p in theOthers:
                    if p in sorted_players_by_trustworthiness[-2:]:
                        return False
            else:
                return False
                #for x in team:
                #    if x in sorted_players_by_trustworthiness[-2:]:
                #        return False
            return True
        else:
            return True
        """
        return True

    def sabotage(self) -> bool:
        """
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
            sus_levels: Dict[TPlayer, float] = self.get_sus_dict()
            sorted_by_sus_level: List[TPlayer] = \
                sorted(self.game.players, key=lambda p1: sus_levels[p1])

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
        """
        return True

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


    def mission_total_suspect_count(self, team: Iterable[TPlayer]) -> int:
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

        team_sus_count: int = self.mission_total_suspect_count(self.game.team)
        if team_sus_count > 5: # capping it at 5
            team_sus_count = 5

        for v in range(0, len(votes)):
            current_player: TPlayer = self.game.players[v]
            if votes[v]:
                self.num_missions_voted_up_with_total_suspect_count[current_player][team_sus_count] += 1
            else:
                self.num_missions_voted_down_with_total_suspect_count[current_player][team_sus_count] += 1

        for p in self.game.players:
            self.training_feature_vectors[p].append(
                [self.game.turn, self.game.tries, p.index, p.name,
                 self.missions_been_on[p], self.failed_missions_been_on[p], self.failed_missions_proposed[p]]
                + self.num_missions_voted_up_with_total_suspect_count[p] +
                self.num_missions_voted_down_with_total_suspect_count[p]
            )

        pass

    def get_sus_dict(self) -> Dict[TPlayer, float]:
        """
        Returns a dictionary with the relative susness of each player
        susness, in this instance, is defined as:
            (failed missions this player has been on + failed missions proposed by player)/
            ((total failed_missions_been_on + total failed proposed missions)/5)
        basically, on average, a player would have a sus level of 1. lower than 1 = legit. higher than 1: very sus
        :return: a dictionary of total_sus_count stuff, calculated with the above formula.
        """
        sus_dict: Dict[TPlayer, float] = {}
        total_sus_count: int = self.mission_total_suspect_count(self.game.players)
        if total_sus_count == 0:
            for p1 in self.game.players:
                sus_dict[p1] = 0
            return sus_dict
        for leader in self.game.players:
            total_sus_count += self.failed_missions_proposed[leader]
        for p in self.game.players:
            sus_dict[p] = (self.failed_missions_been_on[p] + self.failed_missions_proposed[p])/(total_sus_count/5)
        return sus_dict

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
        self.failed_missions_proposed.clear()
        for p in players:
            self.failed_missions_been_on[p] = 0
            self.missions_been_on[p] = 0
            self.failed_missions_proposed[p] = 0
            self.num_missions_voted_up_with_total_suspect_count[p] = [0, 0, 0, 0, 0, 0]
            self.num_missions_voted_down_with_total_suspect_count[p] = [0, 0, 0, 0, 0, 0]

        self.training_feature_vectors.clear()
        for p in players:
            self.training_feature_vectors[p] = []

        # taking note of known spies
        self.spies.clear()
        self.spies = spies.copy()

        pass

    def onMissionComplete(self, sabotaged: int) -> None:
        """Callback once the players have been chosen.
        :param sabotaged:    Integer how many times the mission was sabotaged.
        """

        for p1 in self.game.team:
            self.missions_been_on[p1] += 1

        if sabotaged == 0:
            pass
        else:
            self.failed_missions_proposed[self.game.leader] += 1
            for p2 in self.game.team:
                self.failed_missions_been_on[p2] += 1

        pass

    def onGameComplete(self, win: bool, spies: List[TPlayer]) -> None:
        """Callback once the game is complete, and everything is revealed.
        :param win:          Boolean true if the Resistance won.
        :param spies:        List of only the spies in the game.
        """
        for player_number in range(len(self.game.players)):
            player: TPlayer = self.game.players[player_number]
            spy: bool = player in spies  # This will be a boolean
            feature_vectors = self.training_feature_vectors[player]  # These are our input features
            for v in feature_vectors:
                v.append(
                    1 if spy else 0
                )  # append a 1 or 0 onto the end of our feature vector (for the label, i.e. spy or not spy)
                self.log.debug(','.join(map(str,
                                            v)))  # converts all of elements of v into a csv list, and writes the full csv list to the log file
