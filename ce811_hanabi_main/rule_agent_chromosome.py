# A rule-based hanabi agent driven by a chromosome.
# The objective of this class is to be a starter-class for a larger set of rules.
# M. Fairbank. November 2021.
import random

from hanabi_learning_environment.rl_env import Agent

import enum


from typing import TypedDict, Literal, Union, List, TypeVar, Tuple, Callable, Iterable, Any

Color = Literal["B", "G", "R", "W", "Y"]
CardColor = Literal[None, Color]
Rank = Literal[0, 1, 2, 3, 4]
CardRank = Literal[-1, Rank]
ActionPD = Literal["PLAY", "DISCARD"]
ActionColor = Literal["REVEAL_COLOR"]
ActionRank = Literal["REVEAL_RANK"]

class BaseActionDict(TypedDict):
    pass

class ActionPDDict(BaseActionDict):
    action_type: ActionPD
    card_index: int

class BaseActionRevealDict(BaseActionDict):
    target_offset: int

class ActionColorDict(BaseActionRevealDict):
    action_type: ActionColor
    color: Color

class ActionRankDict(BaseActionRevealDict):
    action_type: ActionRank
    rank: Rank

ActionDict = Union[ActionPDDict, ActionColorDict, ActionRankDict]
ActionType = Literal[ActionPD, ActionColor, ActionRank]

class HandCard(TypedDict):
    color: CardColor
    rank: CardRank

OwnHand = List[HandCard]

class KnownCard(TypedDict):
    color: Color
    rank: Rank

KnownHand = List[KnownCard]

Card = Union[HandCard, KnownCard]

TCard = TypeVar("TCard", bound=Card)

class FireworksDict(TypedDict):
    B: int
    G: int
    R: int
    W: int
    Y: int

class ObservationDict(TypedDict):
    current_player: int
    current_player_offset: int
    deck_size: int
    discard_pile: List[KnownCard]
    fireworks: FireworksDict
    information_tokens: int
    legal_moves: List[ActionDict]
    life_tokens: int
    card_knowledge: List[OwnHand]
    observed_hands: List[Union[OwnHand, KnownHand]]
    num_players: int
    vectorized: List[Literal[0, 1]]



chromo_tuple = Tuple[bool, bool, bool, bool, bool, bool, bool]


class SimpleRuleChromosome(object):

    def __init__(self, config: chromo_tuple = (1, 0, 1, 0, 0, 1, 1)):
        self._chromosome: chromo_tuple = config
        self._fitness: float = SimpleRuleChromosome.fitness_function(self)

    def __lt__(self, other: "SimpleRuleChromosome") -> bool:
        return self.fitness < other._fitness

    @property
    def chromosome(self) -> chromo_tuple:
        return self._chromosome

    @property
    def fitness(self) -> float:
        return self._fitness

    @staticmethod
    def crossover(parent1: "SimpleRuleChromosome", parent2: "SimpleRuleChromosome") -> "SimpleRuleChromosome":
        new_child_list: List[bool] = []
        for i in range(len(parent1.chromosome)):
            if random.random() > 0.5:
                new_child_list.append(parent1.chromosome[i])
            else:
                new_child_list.append(parent2.chromosome[i])
        # noinspection PyTypeChecker
        return SimpleRuleChromosome(tuple(new_child_list))

    @staticmethod
    def mutate(parent: "SimpleRuleChromosome") -> "SimpleRuleChromosome":
        new_child_list: List[bool] = []
        p_len: int = len(parent.chromosome)
        for i in parent.chromosome:
            if random.randrange(0, p_len) == 0:
                new_child_list.append(not i)
            else:
                new_child_list.append(i)
        # noinspection PyTypeChecker
        return SimpleRuleChromosome(tuple(new_child_list))

    @staticmethod
    def fitness_function(individual: "SimpleRuleChromosome") -> float:
        return -1

    @classmethod
    def define_fitness_function(cls, fit_fun: Callable[["SimpleRuleChromosome"], float]):
        SimpleRuleChromosome.fitness_function = fit_fun


    @staticmethod
    def generate_randomly(length: int = 7) -> "SimpleRuleChromosome":
        # noinspection PyTypeChecker
        return SimpleRuleChromosome(tuple(
            random.random() > 0.5 for i in range(0, length)
        ))

    def __str__(self) -> str:
        return "{} fitness {}".format(self.chromosome, self.fitness)

# Best result from above method was actually (False, True, True, True, False, False, False) fitness 16.36


def argmax(llist: List[Any]) -> int:
    #useful function for arg-max
    return llist.index(max(llist))

def argmin(llist: List[Any]) -> int:
    # argmin
    return llist.index(min(llist))
    
class RuleAgentChromosome(Agent):
    """Agent that applies a simple heuristic."""

    def __init__(self, config, chromosome: chromo_tuple, *args, **kwargs):
        # TODO replace this default chromosome with something better, if possible.  Plus, Add new bespoke rules below if necessary.
        """Initialize the agent."""
        self.config = config
        self.chromosome: chromo_tuple = chromosome
        assert isinstance(chromosome, tuple)
        
        # Extract max info tokens or set default to 8.
        self.max_information_tokens = config.get('information_tokens', 8)

    def calculate_all_unseen_cards(self, discard_pile: List[KnownCard], player_hands, fireworks: FireworksDict) -> List[KnownCard]:
        # All of the cards which we can't see are either in our own hand or in the deck.
        # The other cards must be in the discard pile (all cards of which we have seen and remembered) or in other player's hands.
        colors: List[Color] = ['Y', 'B', 'W', 'R', 'G']
        full_hanabi_deck: List[KnownCard] = [{"color":c, "rank":r} for c in colors for r in [0,0,0,1,1,2,2,3,3,4]]
        assert len(full_hanabi_deck)==50 # full hanabi deck size.

        result: List[KnownCard] = full_hanabi_deck.copy()
        # subract off all cards that have been discarded...
        for card in discard_pile:
            if card in result:
                result.remove(card)
        
        # subract off all cards that we can see in the other players' hands...
        for hand in player_hands[1:]:
            for card in hand:
                if card in result:
                    result.remove(card)

        for (color, height) in fireworks.items():
            for rank in range(height):
                card: KnownCard = {"color":color, "rank":rank}
                if card in result:
                    result.remove(card)

        # Now we left with only the cards we have never seen before in the game
        # (so these are the cards in the deck UNION our own hand).
        return result             

    def filter_card_list_by_hint(self, card_list: List[KnownCard], hint: Card) -> List[Card]:
        # This could be enhanced by using negative hint information,
        # available from observation['pyhanabi'].card_knowledge()[player_offset][card_number]
        filtered_card_list: List[Card] = card_list.copy()
        if hint["color"] is not None:
            filtered_card_list = [c for c in filtered_card_list if c["color"] == hint["color"]]
        if hint["rank"] is not None:
            filtered_card_list = [c for c in filtered_card_list if c["rank"] == hint["rank"]]
        return filtered_card_list

    def filter_card_list_by_playability(self, card_list: List[KnownCard], fireworks: FireworksDict) -> List[KnownCard]:
        # find out which cards in card list would fit exactly onto next value of its colour's firework
        return [c for c in card_list if self.is_card_playable(c,fireworks)]

    def filter_card_list_by_unplayable(self, card_list: List[KnownCard], fireworks: FireworksDict) -> List[KnownCard]:
        # find out which cards in card list are always going to be unplayable on its colour's firework
        # This function could be improved by considering that we know a card of value 5 will never be playable
        # if all the 4s for that colour have been discarded.
        return [c for c in card_list if c["rank"] < fireworks[c["color"]]]

    def is_card_playable(self, card: KnownCard, fireworks: FireworksDict) -> bool:
        return card['rank'] == fireworks[card['color']]

    def act(self, observation: ObservationDict) -> Union[ActionDict, None]:
        # this function is called for every player on every turn
        """Act based on an observation."""
        if observation['current_player_offset'] != 0:
            # but only the player with offset 0 is allowed to make an action.  The other players are just observing.
            return None
        
        fireworks: FireworksDict = observation['fireworks']
        card_hints: OwnHand = observation['card_knowledge'][0] # This [0] produces the card hints for OUR own hand (player offset 0)
        hand_size=len(card_hints)

        # build some useful lists of information about what we hold in our hand and what team-mates know about their hands.
        all_unseen_cards: List[KnownCard] = self.calculate_all_unseen_cards(
            observation['discard_pile'], observation['observed_hands'], observation['fireworks']
        )
        possible_cards_by_hand: List[List[Card]] = [self.filter_card_list_by_hint(all_unseen_cards, h) for h in card_hints]
        playable_cards_by_hand: List[List[KnownCard]] =[self.filter_card_list_by_playability(posscards, fireworks) for posscards in possible_cards_by_hand]
        probability_cards_playable: List[float] =[len(playable_cards_by_hand[index])/len(possible_cards_by_hand[index]) for index in range(hand_size)]
        useless_cards_by_hand: List[List[KnownCard]] = [self.filter_card_list_by_unplayable(posscards, fireworks) for posscards in possible_cards_by_hand]
        probability_cards_useless: List[float] =[len(useless_cards_by_hand[index])/len(possible_cards_by_hand[index]) for index in range(hand_size)]
        
        # based on the above calculations, try a sequence of rules in turn and perform the first one that is applicable:
        
        #for rule in self.chromosome:
        if self.chromosome[0] or self.chromosome[1]:
            # Play any highly-probable playable cards:
            threshold=0.8 if self.chromosome[0] else 0.5
            if max(probability_cards_playable) > threshold:
                card_index=argmax(probability_cards_playable)
                return {'action_type': 'PLAY', 'card_index': card_index}

        if self.chromosome[2]:
            # Check if it's possible to hint a card to your colleagues.  TODO this could be split into 2 separate rules?
            if observation['information_tokens'] > 0:
                # Check if there are any playable cards in the hands of the opponents.
                for player_offset in range(1, observation['num_players']):
                    player_hand = observation['observed_hands'][player_offset]
                    player_hints = observation['card_knowledge'][player_offset]
                    # Check if the card in the hand of the opponent is playable.
                    for card, hint in zip(player_hand, player_hints):
                        #if card['rank'] == fireworks[card['color']]:
                        if self.is_card_playable(card, fireworks):
                            if hint['color'] is None:
                                return {
                                    'action_type': 'REVEAL_COLOR',
                                    'color': card['color'],
                                    'target_offset': player_offset
                                }
                            elif hint['rank'] is None:
                                return {
                                    'action_type': 'REVEAL_RANK',
                                    'rank': card['rank'],
                                    'target_offset': player_offset
                                }
        if self.chromosome[3] or self.chromosome[4]:
            # discard any highly-probable useless cards:
            threshold=0.8 if self.chromosome[3] else 0.5
            if observation['information_tokens'] < self.max_information_tokens:
                if max(probability_cards_useless)>threshold:
                    card_index=argmax(probability_cards_useless)
                    return {'action_type': 'DISCARD', 'card_index': card_index}

        if self.chromosome[5]:
            # Discard something
            if observation['information_tokens'] < self.max_information_tokens:
                return {'action_type': 'DISCARD', 'card_index': argmax(probability_cards_useless)}# discards the oldest card (card_index 0 will be oldest card)
        if self.chromosome[6]:
            # Play our best-hope card
            return {'action_type': 'PLAY', 'card_index': argmax(probability_cards_playable)}

        if observation['information_tokens'] < self.max_information_tokens:
            return {'action_type': 'DISCARD', 'card_index': argmax(probability_cards_useless)}
        else:
            return {'action_type': 'PLAY', 'card_index': argmax(probability_cards_playable)}
            # the chromosome contains an unknown rule
            #raise Exception("Rule not defined: "+str(rule))
        # The chromosome needs to be defined so the program never gets to here.  
        # E.g. always include rules 5 and 6 in the chromosome somewhere to ensure this never happens..        
        #raise Exception("No rule fired for game situation - faulty rule set")

