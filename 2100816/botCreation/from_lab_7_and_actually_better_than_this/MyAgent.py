# A rule-based hanabi agent driven by a chromosome.
# The objective of this class is to be a starter-class for a larger set of rules.
# M. Fairbank. November 2021.
import random

from hanabi_learning_environment.rl_env import Agent

from enum import Enum, auto


from typing import TypedDict, Literal, Union, List, TypeVar, Tuple, Callable, Iterable, Any, Dict, Iterator

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


class RulesEnum(Enum):

    PLAY_MOST_PLAYABLE_CARD = 0
    PLAY_MOST_PLAYABLE_CARD_THRESHOLD_HIGH = 1
    PLAY_MOST_PLAYABLE_CARD_THRESHOLD_LOW = 2
    PLAY_MOST_DEFINITELY_PLAYABLE_CARD = 3
    TELL_PLAYER_ABOUT_PLAYABLE_CARD_RANK = 4
    TELL_PLAYER_ABOUT_PLAYABLE_CARD_COLOUR = 5
    TELL_NEXT_PLAYER_ABOUT_PLAYABLE_CARD_RANK = 6
    TELL_NEXT_PLAYER_ABOUT_PLAYABLE_CARD_COLOUR = 7
    TELL_PLAYER_ABOUT_UNPLAYABLE_RANK = 8
    TELL_PLAYER_ABOUT_UNPLAYABLE_COLOR = 9
    TELL_NEXT_PLAYER_ABOUT_UNPLAYABLE_RANK = 10
    TELL_NEXT_PLAYER_ABOUT_UNPLAYABLE_COLOR = 11
    TELL_PLAYER_WITH_MOST_PLAYABLE_CARDS_ABOUT_PLAYABLE_CARDS_RANKS = 12
    TELL_PLAYER_WITH_MOST_PLAYABLE_CARDS_ABOUT_PLAYABLE_CARDS_COLOURS = 13
    DISCARD_MOST_UNPLAYABLE_CARD = 14
    DISCARD_MOST_UNPLAYABLE_CARD_THRESHOLD_HIGH = 15
    DISCARD_MOST_UNPLAYABLE_CARD_THRESHOLD_LOW = 16
    DISCARD_MOST_DEFINITELY_UNPLAYABLE_CARD = 17
    DISCARD_HIGHEST_CARD = 18
    DISCARD_LOWEST_CARD = 19
    DISCARD_OLDEST_CARD = 20
    DISCARD_OLDEST_UNKNOWN_CARD = 21
    TELL_PLAYER_ABOUT_ONES = 22
    TELL_PLAYER_ABOUT_FIVES = 23
    TELL_PLAYER_ABOUT_MOST_COMMON_COLOR = 24
    TELL_PLAYER_ABOUT_LEAST_COMMON_COLOR = 25
    TELL_PLAYER_ABOUT_MOST_PLAYED_COLOR = 26
    TELL_PLAYER_ABOUT_LEAST_PLAYED_COLOR = 27
    TELL_PLAYER_WITH_MOST_USELESS_CARDS_ABOUT_USELESS_RANKS = 28
    TELL_PLAYER_WITH_MOST_USELESS_CARDS_ABOUT_USELESS_COLORS = 29



class RulesChromosome(object):
    rules_list: List[RulesEnum] = [r for r in RulesEnum]

    def __init__(self, config: Union[Tuple[float, ...], List[float], None] = None):

        temp_chromo: List[float] = [
            random.uniform(0.0, 1.0) for r in RulesChromosome.rules_list
        ]
        if config is not None:
            for i in range(min(len(config), len(temp_chromo))):
                temp_chromo[i] = config[i]

        self._chromosome: Tuple[float, ...] = tuple(temp_chromo)

        self._fitness: float = self.fitness_function()

    def __lt__(self, other: "RulesChromosome") -> bool:
        return self.fitness < other.fitness

    @property
    def fitness(self) -> float:
        return self._fitness

    @property
    def to_rule_priority_list(self) -> List[RulesEnum]:
        return sorted(RulesChromosome.rules_list.copy(), key=lambda re: self._chromosome[re.value], reverse=True)

    @property
    def chromosome(self) -> Tuple[float, ...]:
        return self._chromosome

    @staticmethod
    def fitness_function(individual: "RulesChromosome") -> float:
        return -1

    @staticmethod
    def define_fitness_function(fit_fun: Callable[["RulesChromosome"], float]):
        RulesChromosome.fitness_function = fit_fun


    @staticmethod
    def crossover(parent1: "RulesChromosome", parent2: "RulesChromosome") -> "RulesChromosome":
        """Performs crossover via n-dimensional vector interpolation between the values held in the two parents"""
        interp: float = random.random()

        return RulesChromosome(tuple(
            parent1.chromosome[i] + ((parent2.chromosome[i] - parent1.chromosome[i]) * interp)
            for i in range(len(parent1.chromosome))
        ))

    @staticmethod
    def mutate(parent: "RulesChromosome") -> "RulesChromosome":
        """N-dimensional random interpolation"""
        mut_rate: float = 0.25  # 1/len(RulesEnum)
        mutate_vector: List[float, ...] = [
            c + random.uniform(-mut_rate, mut_rate) for c in parent.chromosome
        ]
        for i in range(len(mutate_vector)):
            if mutate_vector[i] > 1:
                mutate_vector[i] -= 1
            elif mutate_vector[i] < 0:
                mutate_vector[i] += 1
        return RulesChromosome(mutate_vector)

    def __str__(self) -> str:
        return "Fitness: {}, Values: {}\nRule order: {}".format(self.fitness, self.chromosome, self.to_rule_priority_list)



# Best result:
# Fitness: 16.06
# Values: (0.4469912448418608, 0.7793399691173142, 0.5883594666862885, 0.03283005840582722, 0.5035798000762269, 0.7331031222972116, 0.07301772405669489, 0.5698950244875348, 0.5769706925880045, 0.37618977340032494, 0.8472024130533544, 0.18300338396894927, 0.6726518968130522, 0.2870108469222508, 0.7260939698695176, 0.3917775357695642, 0.4621767395965047, 0.2894702335044665, 0.49251836358461887, 0.2810412722924867, 0.25894527382791815, 0.6077331417385615, 0.6839671759050361, 0.27827194603099525, 0.028436714207698863, 0.6896115185003819, 0.42260644572052525, 0.3060119374621139, 0.3010773946199542, 0.4532030415409376)
# Rule order: [<RulesEnum.TELL_NEXT_PLAYER_ABOUT_UNPLAYABLE_RANK: 10>, <RulesEnum.PLAY_MOST_PLAYABLE_CARD_THRESHOLD_HIGH: 1>, <RulesEnum.TELL_PLAYER_ABOUT_PLAYABLE_CARD_COLOUR: 5>, <RulesEnum.DISCARD_MOST_UNPLAYABLE_CARD: 14>, <RulesEnum.TELL_PLAYER_ABOUT_LEAST_COMMON_COLOR: 25>, <RulesEnum.TELL_PLAYER_ABOUT_ONES: 22>, <RulesEnum.TELL_PLAYER_WITH_MOST_PLAYABLE_CARDS_ABOUT_PLAYABLE_CARDS_RANKS: 12>, <RulesEnum.DISCARD_OLDEST_UNKNOWN_CARD: 21>, <RulesEnum.PLAY_MOST_PLAYABLE_CARD_THRESHOLD_LOW: 2>, <RulesEnum.TELL_PLAYER_ABOUT_UNPLAYABLE_RANK: 8>, <RulesEnum.TELL_NEXT_PLAYER_ABOUT_PLAYABLE_CARD_COLOUR: 7>, <RulesEnum.TELL_PLAYER_ABOUT_PLAYABLE_CARD_RANK: 4>, <RulesEnum.DISCARD_HIGHEST_CARD: 18>, <RulesEnum.DISCARD_MOST_UNPLAYABLE_CARD_THRESHOLD_LOW: 16>, <RulesEnum.TELL_PLAYER_WITH_MOST_USELESS_CARDS_ABOUT_USELESS_COLORS: 29>, <RulesEnum.PLAY_MOST_PLAYABLE_CARD: 0>, <RulesEnum.TELL_PLAYER_ABOUT_MOST_PLAYED_COLOR: 26>, <RulesEnum.DISCARD_MOST_UNPLAYABLE_CARD_THRESHOLD_HIGH: 15>, <RulesEnum.TELL_PLAYER_ABOUT_UNPLAYABLE_COLOR: 9>, <RulesEnum.TELL_PLAYER_ABOUT_LEAST_PLAYED_COLOR: 27>, <RulesEnum.TELL_PLAYER_WITH_MOST_USELESS_CARDS_ABOUT_USELESS_RANKS: 28>, <RulesEnum.DISCARD_MOST_DEFINITELY_UNPLAYABLE_CARD: 17>, <RulesEnum.TELL_PLAYER_WITH_MOST_PLAYABLE_CARDS_ABOUT_PLAYABLE_CARDS_COLOURS: 13>, <RulesEnum.DISCARD_LOWEST_CARD: 19>, <RulesEnum.TELL_PLAYER_ABOUT_FIVES: 23>, <RulesEnum.DISCARD_OLDEST_CARD: 20>, <RulesEnum.TELL_NEXT_PLAYER_ABOUT_UNPLAYABLE_COLOR: 11>, <RulesEnum.TELL_NEXT_PLAYER_ABOUT_PLAYABLE_CARD_RANK: 6>, <RulesEnum.PLAY_MOST_DEFINITELY_PLAYABLE_CARD: 3>, <RulesEnum.TELL_PLAYER_ABOUT_MOST_COMMON_COLOR: 24>]
best_rule_order_tuple: Tuple[float, ...] = (0.4469912448418608, 0.7793399691173142, 0.5883594666862885, 0.03283005840582722, 0.5035798000762269, 0.7331031222972116, 0.07301772405669489, 0.5698950244875348, 0.5769706925880045, 0.37618977340032494, 0.8472024130533544, 0.18300338396894927, 0.6726518968130522, 0.2870108469222508, 0.7260939698695176, 0.3917775357695642, 0.4621767395965047, 0.2894702335044665, 0.49251836358461887, 0.2810412722924867, 0.25894527382791815, 0.6077331417385615, 0.6839671759050361, 0.27827194603099525, 0.028436714207698863, 0.6896115185003819, 0.42260644572052525, 0.3060119374621139, 0.3010773946199542, 0.4532030415409376)


def generate_premade_rules_order() -> List[RulesEnum]:
    """This loads the 'best' rules order generated by the GA stuff I attempted for lab 7."""
    return sorted([r for r in RulesEnum], key=lambda re: best_rule_order_tuple[re.value], reverse=True)


T = TypeVar("T")


def argmax(llist: List[Any], key=None) -> int:
    #useful function for arg-max
    return llist.index(max(llist, key=key))


def argmin(llist: List[Any], key=None) -> int:
    # argmin
    return llist.index(min(llist, key=key))


class MyAgent(Agent):
    """Agent that applies a simple heuristic."""

    colors: Tuple[Color] = ('Y', 'B', 'W', 'R', 'G')
    ranks: Tuple[int] = (0, 1, 2, 3, 4)
    individual_hanabi_cards: Tuple[KnownCard] = tuple({"color": c, "rank": r} for c in colors for r in
                                         [0, 1, 2, 3, 4])
    full_hanabi_deck: Tuple[KnownCard] = tuple({"color": c, "rank": r} for c in colors for r in
                                         [0, 0, 0, 1, 1, 2, 2, 3, 3, 4])
    individual_cards_and_quantities: Dict[str, int] = dict(
        ("{},{}".format(c["color"],c["rank"]), 3 if c["rank"] == 0 else 1 if c["rank"] == 4 else 2) for c in individual_hanabi_cards
    )

    def __init__(self, config, chromosome: List[RulesEnum]=None, *args, **kwargs):
        """Initialize the agent."""
        self.config = config
        if chromosome is None:
            self.chromosome: List[RulesEnum] = generate_premade_rules_order()
        else:
            self.chromosome: List[RulesEnum] = chromosome
        assert isinstance(self.chromosome, list)
        
        # Extract max info tokens or set default to 8.
        self.max_information_tokens = config.get('information_tokens', 8)

    def calculate_all_unseen_cards(self, discard_pile: List[KnownCard], player_hands: List[List[KnownCard]], fireworks: FireworksDict) -> List[KnownCard]:
        # All of the cards which we can't see are either in our own hand or in the deck.
        # The other cards must be in the discard pile (all cards of which we have seen and remembered) or in other player's hands.
        assert len(MyAgent.full_hanabi_deck)==50 # full hanabi deck size.

        result: List[KnownCard] = list(MyAgent.full_hanabi_deck)
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

    @classmethod
    def get_unplayables_from_discard_pile(cls, discard_pile: List[KnownCard]) -> List[KnownCard]:

        undiscarded_counts: Dict[str, int] = cls.individual_cards_and_quantities.copy()
        for d in discard_pile:
            undiscarded_counts["{},{}".format(d["color"],d["rank"])] -= 1

        discard_unplayables: List[KnownCard] = []
        for card in (nd[0] for nd in undiscarded_counts.items() if nd[1] == 0):
            c_list = card.split(",")
            try:
                c_rank: int = int(c_list[1])
                if {"color":c_list[0], "rank":c_rank} in discard_unplayables:
                    continue
                elif c_rank < 4:
                    current_h = c_rank + 1
                    while current_h <= 4:
                        # noinspection PyTypeChecker
                        discard_unplayables.append({"color":c_list[0], "rank":current_h})
                        current_h += 1
            except ValueError:
                pass
        return discard_unplayables


    def filter_card_list_by_unplayable(self, card_list: List[KnownCard], fireworks: FireworksDict, discard_unplayable: List[KnownCard]) -> List[KnownCard]:
        # find out which cards in card list are always going to be unplayable on its colour's firework
        # This function could be improved by considering that we know a card of value 5 will never be playable
        # if all the 4s for that colour have been discarded.
        return [c for c in card_list if c["rank"] < fireworks[c["color"]] and c not in discard_unplayable]



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

        discarded: List[KnownCard] = observation['discard_pile']

        discard_unplayables: List[KnownCard] = MyAgent.get_unplayables_from_discard_pile(discarded)

        # build some useful lists of information about what we hold in our hand and what team-mates know about their hands.
        all_unseen_cards: List[KnownCard] = self.calculate_all_unseen_cards(
            discarded, observation['observed_hands'], fireworks
        )
        possible_cards_by_hand: List[List[Card]] = [self.filter_card_list_by_hint(all_unseen_cards, h) for h in card_hints]
        playable_cards_by_hand: List[List[KnownCard]] =[self.filter_card_list_by_playability(posscards, fireworks) for posscards in possible_cards_by_hand]
        probability_cards_playable: List[float] =[len(playable_cards_by_hand[index])/len(possible_cards_by_hand[index]) for index in range(hand_size)]
        useless_cards_by_hand: List[List[KnownCard]] = [self.filter_card_list_by_unplayable(posscards, fireworks, discard_unplayables) for posscards in possible_cards_by_hand]
        probability_cards_useless: List[float] =[len(useless_cards_by_hand[index])/len(possible_cards_by_hand[index]) for index in range(hand_size)]

        other_player_info = TypedDict("other_player_info", {"playable": List[KnownCard],"useless": List[KnownCard],
                                                            "unknown ranks": List[KnownCard], "unknown colors": List[KnownCard]})
        others_info: Dict[int, other_player_info] = {}
        for i in range(1, observation['num_players']):
            other_cards: List[KnownCard] = observation['observed_hands'][i]
            other_hand: List[HandCard] = observation['card_knowledge'][i]
            others_info[i] = {
                "playable": self.filter_card_list_by_playability(other_cards, fireworks),
                "useless": self.filter_card_list_by_unplayable(other_cards, fireworks, discard_unplayables),
                "unknown ranks": [other_cards[i] for i in range(len(other_cards)) if other_hand[i]["rank"] is None],
                "unknown colors": [other_cards[i] for i in range(len(other_cards)) if other_hand[i]["color"] is None]
            }

        my_unknown_cards: List[Card] = [c for c in card_hints if c["rank"] is None or c["color"] is None]
        my_known_cards: List[Card] = [c for c in card_hints if c["rank"] is not None or c["color"] is not None]
        my_known_ranks: List[Card] = [c for c in my_known_cards if c["rank"]  is not None]
        my_known_colors:List[Card] = [c for c in my_known_cards if c["color"] is not None]

        # based on the above calculations, try a sequence of rules in turn and perform the first one that is applicable:

        can_discard: bool = observation['information_tokens'] < self.max_information_tokens

        can_inform: bool = observation['information_tokens'] > 0

        # noinspection PyTypeChecker
        most_played_colours: List[Color] = [kv[0] for kv in fireworks.items() if kv[1] == max(fireworks.values())]
        # noinspection PyTypeChecker
        least_played_colors: List[Color] = [kv[0] for kv in fireworks.items() if kv[1] == min(fireworks.values())]

        for rule in self.chromosome:

            if rule == RulesEnum.PLAY_MOST_PLAYABLE_CARD:
                return {'action_type': 'PLAY', 'card_index': argmax(probability_cards_playable)}
            elif rule == RulesEnum.PLAY_MOST_DEFINITELY_PLAYABLE_CARD:
                if max(probability_cards_playable) == 1:
                    return {'action_type': 'PLAY', 'card_index': argmax(probability_cards_playable)}
            elif rule == RulesEnum.PLAY_MOST_PLAYABLE_CARD_THRESHOLD_HIGH:
                if max(probability_cards_playable) > 0.8:
                    return {'action_type': 'PLAY', 'card_index': argmax(probability_cards_playable)}
            elif rule == RulesEnum.PLAY_MOST_PLAYABLE_CARD_THRESHOLD_LOW:
                if max(probability_cards_playable) > 0.5:
                    return {'action_type': 'PLAY', 'card_index': argmax(probability_cards_playable)}

            if can_discard:
                if rule == RulesEnum.DISCARD_HIGHEST_CARD:
                    if len(my_known_ranks) > 0:
                        return {'action_type': 'DISCARD', 'card_index': argmax(card_hints, lambda c: c["rank"] if c["rank"] is not None else -1)}
                elif rule == RulesEnum.DISCARD_LOWEST_CARD:
                    if len(my_known_ranks) > 0:
                        return {'action_type': 'DISCARD', 'card_index': argmin(card_hints, lambda c: c["rank"] if c["rank"] is not None else 10)}
                elif rule == RulesEnum.DISCARD_MOST_UNPLAYABLE_CARD:
                    return {'action_type': 'DISCARD', 'card_index': argmax(probability_cards_useless)}
                elif rule == RulesEnum.DISCARD_MOST_DEFINITELY_UNPLAYABLE_CARD:
                    if max(probability_cards_useless) == 1:
                        return {'action_type': 'DISCARD', 'card_index': argmax(probability_cards_useless)}
                elif rule == RulesEnum.DISCARD_MOST_UNPLAYABLE_CARD_THRESHOLD_HIGH:
                    if max(probability_cards_useless) > 0.8:
                        return {'action_type': 'DISCARD', 'card_index': argmax(probability_cards_useless)}
                elif rule == RulesEnum.DISCARD_MOST_UNPLAYABLE_CARD_THRESHOLD_LOW:
                    if max(probability_cards_useless) > 0.5:
                        return {'action_type': 'DISCARD', 'card_index': argmax(probability_cards_useless)}
                elif rule == RulesEnum.DISCARD_OLDEST_CARD:
                    return {'action_type': 'DISCARD', 'card_index': 0}
                elif rule == RulesEnum.DISCARD_OLDEST_UNKNOWN_CARD:
                    if len(my_unknown_cards) > 0:
                        return {'action_type': 'DISCARD', 'card_index': card_hints.index(my_unknown_cards[0])}

            if can_inform:
                if rule == RulesEnum.TELL_PLAYER_ABOUT_ONES:
                    for i in range(1, observation['num_players']):
                        if any((others_info[i]["unknown ranks"][c]["rank"] == 0 for c in range(len(others_info[i]["unknown ranks"])))):
                            return {
                                'action_type': 'REVEAL_RANK',
                                'rank': 0,
                                'target_offset': i
                            }
                elif rule == RulesEnum.TELL_PLAYER_ABOUT_FIVES:
                    for i in range(1, observation['num_players']):
                        if any((others_info[i]["unknown ranks"][c]["rank"] == 4 for c in range(len(others_info[i]["unknown ranks"])))):
                            return {
                                'action_type': 'REVEAL_RANK',
                                'rank': 4,
                                'target_offset': i
                            }
                elif rule == RulesEnum.TELL_PLAYER_ABOUT_PLAYABLE_CARD_COLOUR or rule == RulesEnum.TELL_NEXT_PLAYER_ABOUT_PLAYABLE_CARD_COLOUR:
                    for i in range(1, observation['num_players']):
                        for c in others_info[i]["unknown colors"]:
                            if self.is_card_playable(c, fireworks):
                                return {
                                    'action_type': 'REVEAL_COLOR',
                                    'color': c["color"],
                                    'target_offset': i
                                }
                        if rule == RulesEnum.TELL_NEXT_PLAYER_ABOUT_PLAYABLE_CARD_COLOUR:
                            break
                elif rule == RulesEnum.TELL_PLAYER_ABOUT_PLAYABLE_CARD_RANK or rule == RulesEnum.TELL_NEXT_PLAYER_ABOUT_PLAYABLE_CARD_RANK:
                    for i in range(1, observation['num_players']):
                        for c in others_info[i]["unknown ranks"]:
                            if self.is_card_playable(c, fireworks):
                                return {
                                    'action_type': 'REVEAL_RANK',
                                    'rank': c["rank"],
                                    'target_offset': i
                                }
                        if rule == RulesEnum.TELL_PLAYER_ABOUT_PLAYABLE_CARD_RANK:
                            break
                elif rule == RulesEnum.TELL_PLAYER_ABOUT_UNPLAYABLE_COLOR or rule == RulesEnum.TELL_NEXT_PLAYER_ABOUT_UNPLAYABLE_COLOR:
                    for i in range(1, observation['num_players']):
                        for c in others_info[i]["unknown colors"]:
                            if not self.is_card_playable(c, fireworks):
                                return {
                                    'action_type': 'REVEAL_COLOR',
                                    'color': c["color"],
                                    'target_offset': i
                                }
                        if rule == RulesEnum.TELL_NEXT_PLAYER_ABOUT_UNPLAYABLE_COLOR:
                            break
                elif rule == RulesEnum.TELL_PLAYER_ABOUT_UNPLAYABLE_RANK or rule == RulesEnum.TELL_NEXT_PLAYER_ABOUT_UNPLAYABLE_RANK:
                    for i in range(1, observation['num_players']):
                        for c in others_info[i]["unknown ranks"]:
                            if self.is_card_playable(c, fireworks):
                                return {
                                    'action_type': 'REVEAL_RANK',
                                    'rank': c["rank"],
                                    'target_offset': i
                                }
                        if rule == RulesEnum.TELL_NEXT_PLAYER_ABOUT_UNPLAYABLE_RANK:
                            break
                elif rule == RulesEnum.TELL_PLAYER_ABOUT_MOST_PLAYED_COLOR:
                    for i in range(1, observation['num_players']):
                        for c in others_info[i]["unknown colors"]:
                            if c["color"] in most_played_colours:
                                return {
                                    'action_type': 'REVEAL_COLOR',
                                    'color': c["color"],
                                    'target_offset': i
                                }
                elif rule == RulesEnum.TELL_PLAYER_ABOUT_LEAST_PLAYED_COLOR:
                    for i in range(1, observation['num_players']):
                        for c in others_info[i]["unknown colors"]:
                            if c["color"] in least_played_colors:
                                return {
                                    'action_type': 'REVEAL_COLOR',
                                    'color': c["color"],
                                    'target_offset': i
                                }
                elif rule == RulesEnum.TELL_PLAYER_ABOUT_MOST_COMMON_COLOR or rule == RulesEnum.TELL_PLAYER_ABOUT_LEAST_COMMON_COLOR:
                    for i in range(1, observation['num_players']):
                        if len(others_info[i]["unknown colors"]) == 0:
                            continue
                        else:
                            col_counts: Dict[Color, int] = {}
                            for c in others_info[i]["unknown colors"]:
                                if c["color"] not in col_counts.keys():
                                    col_counts[c["color"]] = 0
                                col_counts[c["color"]] += 1
                            if rule == RulesEnum.TELL_PLAYER_ABOUT_MOST_COMMON_COLOR:
                                return {
                                    'action_type': 'REVEAL_COLOR',
                                    'color': max(col_counts.items(), key=lambda kv:kv[1])[0],
                                    'target_offset': i
                                }
                            else:
                                return {
                                    'action_type': 'REVEAL_COLOR',
                                    'color': min(col_counts.items(), key=lambda kv: kv[1])[0],
                                    'target_offset': i
                                }
                elif rule == RulesEnum.TELL_PLAYER_WITH_MOST_USELESS_CARDS_ABOUT_USELESS_COLORS or rule == RulesEnum.TELL_PLAYER_WITH_MOST_USELESS_CARDS_ABOUT_USELESS_RANKS:
                    sorted_by_useless: List[Tuple[int, other_player_info]] = [*others_info.items()]
                    sorted_by_useless.sort(key=lambda kv: len(kv[1]["useless"]))
                    if rule == RulesEnum.TELL_PLAYER_WITH_MOST_USELESS_CARDS_ABOUT_USELESS_COLORS:
                        for kv in sorted_by_useless:
                            unknown_useless_c = tuple(c for c in kv[1]["useless"] if c in kv[1]["unknown colors"])
                            if len(unknown_useless_c) > 0:
                                return {
                                    'action_type': 'REVEAL_COLOR',
                                    'color': unknown_useless_c[0]["color"],
                                    'target_offset': kv[0]
                                }
                    else:
                        for kv in sorted_by_useless:
                            unknown_useless_r = tuple(c for c in kv[1]["useless"] if c in kv[1]["unknown ranks"])
                            if len(unknown_useless_r) > 0:
                                return {
                                    'action_type': 'REVEAL_RANK',
                                    'rank': unknown_useless_r[0]["rank"],
                                    'target_offset': kv[0]
                                }

        if observation['information_tokens'] < self.max_information_tokens:
            return {'action_type': 'DISCARD', 'card_index': argmax(probability_cards_useless)}
        else:
            return {'action_type': 'PLAY', 'card_index': argmax(probability_cards_playable)}
            # the chromosome contains an unknown rule
            #raise Exception("Rule not defined: "+str(rule))
        # The chromosome needs to be defined so the program never gets to here.  
        # E.g. always include rules 5 and 6 in the chromosome somewhere to ensure this never happens..        
        #raise Exception("No rule fired for game situation - faulty rule set")

