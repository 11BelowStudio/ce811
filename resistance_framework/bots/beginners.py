# All of the example bots in this file derive from the base Bot class.  See
# how this is implemented by looking at player.py.  The API is very well
# documented there.
from player import Bot, TPlayer # TPlayer is a type annotation for any subtype of the 'Player' class.

# Each bot has access to the game state, stored in the self.game member
# variable.  See the State class in game.py for the full list of variables you
# have access to from your bot.
# 
# The examples below purposefully use only self.game to emphasize its
# importance.  Advanced bots tend to only use the game State class to decide!
from game import State


# Many bots will use random decisions to break ties between two equally valid
# options.  The simple bots below rely on randomness heavily, and expert bots
# tend to use other statistics and criteria (e.g. who is winning) to avoid ties
# altogether!
import random


class Paranoid(Bot):
    """An AI bot that tends to vote everything down!"""

    def select(self, players, count):
        self.say("Picking myself and others I don't trust.")
        return [self] + random.sample(self.others(), count - 1)

    def vote(self, team):
        if (self.spy == False):
            if (self in team):
                self.say("I'm a resistance member on the team, so I approve!")
                return True
            if (self.game.tries == 5):
                # TODO: where the fuck is the actual 'base game tries' variable???
                self.say("I don't like this, but its the last try, so I have to approve it.")
                return True

        self.say("I only vote for my own missions.")
        return bool(self == self.game.leader)

    def sabotage(self):

        if (self.game.turn == 1):
            self.log.debug("It's turn 1, no sabotaging today!")
            return False
        if (len(self.game.team) == 2):
            self.log.debug("There's only two of us, too risky for me to sabotage it")
            return False

        self.log.debug("I always sabotage when I'm a spy when it isn't turn 1 and there's more than 2 people here")
        return True


class CountingBot(Bot):
    """An AI bot that counts stuff"""

    def __init__(self, game: State, index: int, spy: bool):
        super(CountingBot, self).__init__(game, index, spy)
        self.game: State = game

        self.failed_missions_been_on: dict[TPlayer, int] = {}
        """ A dictionary that keeps count of how many times each player has been on a team that failed."""


    def onGameRevealed(self, players: list[TPlayer], spies: list[TPlayer]) -> None:
        """
        At the start of each new game, we clear the list of failed missions that each player has been on,
        and repopulate it with each player's failed mission count reset to 0
        :param players: the list of the players playing this game
        :param spies: the list of spies this game (only visible if this player is a spy)
        :return: nothing.
        """
        self.failed_missions_been_on.clear()
        for p in players:
            self.failed_missions_been_on[p] = 0

    def onMissionComplete(self, sabotaged: int) -> None:
        """
        At the end of a mission, we see if it was sabotaged (and, if it was, we increment the count of
        failed missions for the players on the team appropriately).

        Also, if every single player on the team sabotaged it, they're all sus af, so they all
        get a 'failed missions been on' of a stupidly high number (because none of them are free of sin)
        :param sabotaged: how many players sabotaged the mission
        :return: nothing
        """
        if sabotaged == 0:
            pass
        else:
            if len(self.game.team) == sabotaged:
                for k in self.game.team:
                    self.failed_missions_been_on[k] = 9999999999999999999999999
            else:
                for p in self.game.team:
                    self.failed_missions_been_on[p] += 1

    def select(self, players: list[TPlayer], count: int) -> list[TPlayer]:
        """
        Chooses this player,
        and fills the other slots with the other player(s) who have been on the fewest failed missions
        :param players: all the players
        :param count: how many players to select
        :return: a list with this player and the other players who are least untrustworthy
        """
        self.say("Picking myself and others I distrust the least.")

        #sorted_players: list[TPlayer] = sorted(self.others(), key=lambda p: self.failed_missions_been_on[p])[:count-1]
        #return [self] + sorted_players[:count-1]
        return [self] + sorted(self.others(), key=lambda p: self.failed_missions_been_on[p])[:count-1]

    def vote(self, team: list[TPlayer]) -> bool:
        """
        Opposes all missions if this player is a spy.
        Otherwise, opposes all missions that contain the two players with the highest count of failed missions.
        :param team: the team being voted on
        :return:
        """
        if self.spy:
            return False

        most_sus: set[TPlayer] = set(sorted(self.others(), key=lambda x: self.failed_missions_been_on[x])[-2:])
        """ The two most suspicious players (most failed missions) """

        if most_sus.isdisjoint(team): # vote yes if those two are not in the mission
            return True
        else:
            return False

    def sabotage(self):

        if self.game.turn == 1:
            self.log.debug("It's turn 1, no sabotaging today!")
            return False
        if len(self.game.team) == 2:
            self.log.debug("There's only two of us, too risky for me to sabotage it")
            return False

        self.log.debug("I always sabotage when I'm a spy when it isn't turn 1 and there's more than 2 people here")
        return True



class Hippie(Bot):
    """An AI bot that's OK with everything!"""

    def select(self, players, count):
        self.say("Picking some cool dudes to go with me!")
        return [self] + random.sample(self.others(), count - 1)

    def vote(self, team): 
        self.say("Everything is OK with me, man.")
        return True

    def sabotage(self):
        self.log.debug("Sabotaging is what spy dudes do, right?")
        return True


class RandomBot(Bot):
    """An AI bot that's perhaps never played before and doesn't understand the
    rules very well!"""

    def select(self, players, count):
        self.say("A completely random selection.")
        return random.sample(self.game.players, count)

    def vote(self, team): 
        self.say("A completely random vote.")
        return random.choice([True, False])

    def sabotage(self):
        self.log.debug("A completely random sabotage.")
        return random.choice([True, False])

    def announce(self):
        subset = random.sample(self.others(), random.randint(0, len(self.others())))
        return {p: random.random() for p in subset}


class Neighbor(Bot):
    """An AI that picks and votes for its neighbours and specifically does not
    use randomness in its decision-making."""

    @property
    def neighbors(self):
        n = self.game.players[self.index:] + self.game.players[0:self.index]
        return n

    def select(self, players, count):
        return self.neighbors[0:count]

    def vote(self, team):
        if self.game.tries == 5:
            return not self.spy
        n = self.neighbors[0:len(team)] + [self]
        for p in team:
            if not p in n: return False
        return True

    def sabotage(self):
        return len(self.game.team) == 2 or self.game.turn > 3


class Deceiver(Bot):
    """A tricky bot that's good at pretending being resistance as a spy."""

    def onGameRevealed(self, players, spies):
        self.spies = spies

    def select(self, players, count):
        return [self] + random.sample(self.others(), count - 1)

    def vote(self, team): 
        # Since a resistance would vote up the last mission...
        if self.game.tries == 5:
            return True
        # Spies select any mission with only one spy on it.
        if self.spy and len(self.game.team) == 2:
            return len([p for p in self.game.team if p in self.spies]) == 1
        # If I'm not on the team, and it's a team of 3...
        if len(self.game.team) == 3 and not self in self.game.team: 
            return False
        return True

    def sabotage(self):
        # Shoot down only missions with more than another person.
        return len(self.game.team) > 2


class RuleFollower(Bot):
    """Rule-based AI that does a pretty good job of capturing
    common sense play rules for THE RESISTANCE."""

    def onGameRevealed(self, players, spies):
        self.spies = spies

    def select(self, players, count):
        return [self] + random.sample(self.others(), count - 1)

    def vote(self, team): 
        # Both types of factions have constant behavior on the last try.
        if self.game.tries == 5:
            return not self.spy
        # Spies select any mission with one or more spies on it.
        if self.spy:
            return len([p for p in self.game.team if p in self.spies]) > 0
        # If I'm not on the team, and it's a team of 3...
        if len(self.game.team) == 3 and not self in self.game.team:
            return False
        return True

    def sabotage(self):
        return True


class Jammer(Bot):
    """An AI bot that plays simply as Resistance, but as a Spy plays against
    the common wisdom for synchronizing sabotages."""

    def onGameRevealed(self, players, spies):
        self.spies = spies

    def select(self, players, count):
        if not self.spies:
            return random.sample(self.game.players, count)
        else:
            # Purposefully go out of our way to pick the other spy so that we
            # can trick him with deceptive sabotaging!
            self.log.info("Picking the other spy to trick them!")    
            return list(self.spies) + random.sample(set(self.game.players) - set(self.spies), count-2)

    def vote(self, team): 
        return True

    def sabotage(self):
        spies = [s for s in self.game.team if s in self.spies]
        if len(spies) > 1:
            # Intermediate to advanced bots assume that sabotage is "controlled"
            # by the mission leader, so we go against this practice here.
            if self == self.game.leader:
                self.log.info("Not coordinating not sabotaging because I'm leader.")
                return False 

            # This is the opposite of the same practice, sabotage if the other
            # bot is expecting "control" the sabotage.
            if self.game.leader in spies:
                self.log.info("Not coordinating and sabotaging despite the other spy being leader.")
                return True
            spies.remove(self)

            # Often, intermeditae bots synchronize based on their global index
            # number.  Here we go against the standard pracitce and do it the
            # other way around!
            self.log.info("Coordinating according to the position around the table...")
            return self.index > spies[0].index
        return True




