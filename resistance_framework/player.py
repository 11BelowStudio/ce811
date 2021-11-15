import logging
import logging.handlers

import core

from typing import TypeVar, List, Dict, Set, NoReturn

TPlayer = TypeVar("TPlayer", bound="Player")
"""Anything that's a subtype of the 'Player' class"""


class Player(object):
    """A player in the game of resistance, identified by a unique index as the
       position at the table (random), and a name that identifies this player
       across multiple games (constant).
       
       When you build a Bot for resistance, you'll be given lists of players to
       manipulate in the form of instances of this Player class.  You can use
       it as follows:
        
            for player in players:
                print player.name, player.index

       NOTE: You can ignore the implementation of this class and simply skip to
       the details of the Bot class below if you want to write your own AI.
    """

    def __init__(self, name: str, index: int):
        # Setup the two member variables first, then continue...
        self.name: str = name
        self.index: int = index
        # This line is necessary for bots using mods as mix-in classes.
        super(Player, self).__init__()

    def __repr__(self):
        return "%i-%s" % (self.index, self.name)

    def __eq__(self, other):
        return self.index == other.index and self.name == other.name

    def __le__(self, other):
        return self.index < other.index

    def __ne__(self, other):
        return self.index != other.index or self.name != other.name

    def __hash__(self):
        return hash(self.index) ^ hash(self.name)


# noinspection PyPep8
from game import State


class Bot(Player):
    """This is the base class for your AI in THE RESISTANCE.  To get started:
         1) Derive this class from a new file that will contain your AI.  See
            bots.py for simple stock AI examples.

         2) Implement mandatory API functions below; you must re-implement
            those that raise exceptions (i.e. vote, select, sabotage).

         3) If you need any of the optional callback API functions, implement
            them (i.e. all functions named on*() are callbacks).

       Aside from parameters passed as arguments to the functions below, you 
       can also access the game state via the self.game variable, which contains
       a State class defined in game.py.

       For debugging, it's recommended you use the self.log variable, which
       contains a python logging object on which you can call .info() .debug()
       or warn() for instance.  The output is stored in a file in the #/logs/
       folder, named according to your bot. 
    """

    __metaclass__ = core.Observable

    def onGameRevealed(self, players: List[TPlayer], spies: Set[TPlayer]) -> NoReturn:
        """This function will be called to list all the players, and if you're
        a spy, the spies too -- including others and yourself.
        :param players:  List of all players in the game including you.
        :param spies:    Set of players that are spies (if you are a spy), or an empty set (if you aren't a spy).
        """
        pass

    def onMissionAttempt(self, mission: int, tries: int, leader: TPlayer) -> NoReturn:
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
        raise NotImplemented

    def onTeamSelected(self, leader: TPlayer, team: List[TPlayer]) -> NoReturn:
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
        raise NotImplemented

    def onVoteComplete(self, votes: List[bool]) -> NoReturn:
        """Callback once the whole team has voted.
        :param: votes        Boolean votes for each player (ordered).
        """
        pass

    def sabotage(self) -> bool:
        """Decide what to do on the mission once it has been approved.  This
        function is only called if you're a spy, otherwise you have no choice.
        :return: bool        Yes to shoot down a mission.
        """
        raise NotImplemented

    def onMissionComplete(self, sabotaged: int) -> NoReturn:
        """Callback once the players have been chosen.
        :param sabotaged:    Integer how many times the mission was sabotaged.
        """
        pass

    def onMissionFailed(self, leader: TPlayer, team: List[TPlayer]) -> NoReturn:
        """Callback once a vote did not reach majority, failing the mission.
        :param leader:       The player responsible for selection.
        :param team:         The list of players chosen for the mission.
        """
        pass

    def announce(self) -> Dict[Player, float]:
        """Publicly state beliefs about the game's state by announcing spy
        probabilities for any combination of players in the game.  This is
        done after each mission completes, and takes the form of a mapping from
        player to float.  Not all players must be specified, and of course this
        can be innacurate!

        :return: Dict[Player, float]     Mapping of player to spy probability.
        """
        return {}

    def onAnnouncement(self, source: Player, announcement: Dict[Player, float]) -> NoReturn:
        """Callback if another player decides to announce beliefs about the
        game.  This is passed as a potentially incomplete mapping from player
        to spy probability.

        :param source:        Player making the announcement.
        :param announcement:  Dictionary mapping players to spy probabilities.
        """
        pass

    def say(self, message: str) -> NoReturn:
        """Helper function to print a message in the global game chat, visible
        by all the other players.

        :param message:       String containing free-form text.
        """
        self.log.info(message)

    def onMessage(self, source: TPlayer, message: str) -> NoReturn:
        """Callback if another player sends a general free-form message to the
        channel.  This is passed in as a generic string that needs to be parsed.

        :param source:       Player sending the message.
        :param message:  Arbitrary string for the message sent.
        """
        pass

    def onGameComplete(self, win: bool, spies: Set[TPlayer]) -> NoReturn:
        """Callback once the game is complete, and everything is revealed.
        :param win:          Boolean true if the Resistance won.
        :param spies:        List of only the spies in the game.
        """
        pass

    def others(self) -> List[TPlayer]:
        """Helper function to list players in the game that are not your bot."""
        return [p for p in self.game.players if p != self]

    def __init__(self, game: State, index: int, spy: bool):
        """Constructor called before a game starts.  It's recommended you don't
        override this function and instead use onGameRevealed() to perform
        setup for your AI.
        :param game:     the current game state
        :param index:    Your own index in the player list.
        :param spy:      Are you supposed to play as a spy?
        """
        super(Bot, self).__init__(self.__class__.__name__, index)

        self.game: State = game
        """ The current gamestate """
        self.spy: bool = spy
        """ Whether or not this player is a spy"""

        self.log = logging.getLogger(self.name)
        if not self.log.handlers:
            try:
                output = logging.FileHandler(filename='logs/'+self.name+'.log')
                self.log.addHandler(output)
                self.log.setLevel(logging.DEBUG)
            except IOError:
                pass

    def __repr__(self) -> str:
        """Built-in function to support pretty-printing."""
        t = {True: "SPY", False: "RST"}
        return "<%s #%i %s>" % (self.name, self.index, t[self.spy])

