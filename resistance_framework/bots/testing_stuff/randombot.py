import random

from player import Bot, TPlayer
from typing import List


class RandomBot(Bot):

    def select(self, players: List[TPlayer], count: int) -> List[TPlayer]:
        return random.sample(players, count)
    def vote(self, team: List[TPlayer]) -> bool:
        return random.random() > 0.5
    def sabotage(self) -> bool:
        return random.random() > 0.5
