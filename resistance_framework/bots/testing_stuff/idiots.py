
from player import Bot, TPlayer
from typing import List

g_spies: List[TPlayer]

class DumbBot(Bot):

    def select(self, players: List[TPlayer], count: int) -> List[TPlayer]:
        if self.spy and count == 2:
            global g_spies
            return g_spies
        return [self] + self.others()[:count-1]
    def vote(self, team: List[TPlayer]) -> bool:
        return True
    def sabotage(self) -> bool:
        return True
    def onGameRevealed(self, players: List[TPlayer], spies: List[TPlayer]) -> None:
        if len(spies) > 0:
            global g_spies
            g_spies = spies
        pass


class DumbBot2(Bot):
    def select(self, players: List[TPlayer], count: int) -> List[TPlayer]:
        if self.spy and count == 2:
            global g_spies
            return g_spies
        return [self] + self.others()[:count - 1]
    def vote(self, team: List[TPlayer]) -> bool:
        return True
    def sabotage(self) -> bool:
        return True
    def onGameRevealed(self, players: List[TPlayer], spies: List[TPlayer]) -> None:
        if len(spies) > 0:
            global g_spies
            g_spies = spies
        pass


class DumbBot3(Bot):
    def select(self, players: List[TPlayer], count: int) -> List[TPlayer]:
        if self.spy and count == 2:
            global g_spies
            return g_spies
        return [self] + self.others()[:count - 1]
    def vote(self, team: List[TPlayer]) -> bool:
        return True
    def sabotage(self) -> bool:
        return True
    def onGameRevealed(self, players: List[TPlayer], spies: List[TPlayer]) -> None:
        if len(spies) > 0:
            global g_spies
            g_spies = spies
        pass


class DumbBot4(Bot):
    def select(self, players: List[TPlayer], count: int) -> List[TPlayer]:
        if self.spy and count == 2:
            global g_spies
            return g_spies
        return [self] + self.others()[:count - 1]
    def vote(self, team: List[TPlayer]) -> bool:
        return True
    def sabotage(self) -> bool:
        return True
    def onGameRevealed(self, players: List[TPlayer], spies: List[TPlayer]) -> None:
        if len(spies) > 0:
            global g_spies
            g_spies = spies
        pass