
from player import Bot, Player
from game import State

from typing import TypeVar, List, Dict, Set

TPlayer = TypeVar("TPlayer", bound="Player")

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

        First step: probably trying to get some sort of bayesian classifier working.
        10 possible role combinations (6 of which need to be worked out if resistance),
        need to think of how to store a bayesian belief network into a data structure,
        and how to get code to use it.

        Might need to take another look at the statistician bot, because that appears to use a naive bayes classifier.
        Or look at exising Python implementations of it for later on.

        """
        super().__init__(game, index, spy)

        


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



