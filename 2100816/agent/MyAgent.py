#%%
import math
import random
import warnings

from hanabi_learning_environment.rl_env import Agent
import numpy as np
from enum import Enum, auto, Flag, IntFlag
import dataclasses
from dataclasses import dataclass
import functools
import itertools
import abc
from collections import defaultdict
import pprint
import queue
import collections

from hanabi_learning_environment import rl_env
from hanabi_learning_environment.rl_env import Agent

from scipy import signal

from typing import NoReturn, Callable, Sequence, Final, Protocol, TypedDict, Literal, Union, ClassVar, List, FrozenSet, Generic, TypeVar, Set, Tuple, Callable, Iterable, Any, Dict, Iterator, Optional


rng: np.random.Generator = np.random.default_rng()

#%%

Color = Literal["B", "G", "R", "W", "Y"]
CardColor = Optional[Color]
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

def actplay(ind: int) -> ActionPDDict:
    """
    Helper method to create the appropriate dict for playing a card
    :param ind: index of card to play
    :return: an actionPDDict that plays that specified card
    """
    return {
        "action_type": "PLAY",
        "card_index": ind
    }

def actdiscard(ind: int) -> ActionPDDict:
    """
    Helper method for discarding
    :param ind: index of card being discarded
    :return: the appropriate actionPDDict
    """
    return {
        "action_type": "DISCARD",
        "card_index": ind
    }

def actcolor(offset: Union[int, "OtherInfo"], colour: Color) -> ActionColorDict:
    """
    Helper method for revealing colour
    :param offset: player to target
    :param colour: colour to reveal
    :return: the appropriate ActionColorDict
    """
    if isinstance(offset, OtherInfo):
        offset = offset.offset
    return {
        "action_type": "REVEAL_COLOR",
        "color": colour,
        "target_offset": offset
    }

class ActionRankDict(BaseActionRevealDict):
    action_type: ActionRank
    rank: Rank

def actrank(offset: Union[int, "OtherInfo"], rank: Rank) -> ActionRankDict:
    """
    Helper method for revealing rank
    :param offset: player to target
    :param rank: rank to reveal
    :return: the appropriate ActionRankDict
    """
    if isinstance(offset, OtherInfo):
        offset = offset.offset
    return {
        "action_type": "REVEAL_RANK",
        "rank": rank,
        "target_offset": offset
    }

ActionDict = Union[ActionPDDict, ActionColorDict, ActionRankDict]
Action = Union[ActionDict, int]

class HandCard(TypedDict):
    color: CardColor
    rank: CardRank

OwnHand = List[HandCard]

class KnownCard(TypedDict):
    color: Color
    rank: Rank

KnownHand = List[KnownCard]

Card = Union[HandCard, KnownCard]


@dataclass(init=False, repr=True, eq=True, frozen=True)
class CardData:
    color: CardColor
    rank: CardRank

    UNKNOWN_CARD: ClassVar["CardData"]
    VALID_RANKS: ClassVar[FrozenSet[Rank]] = frozenset((0,1,2,3,4))
    VALID_COLOR: ClassVar[FrozenSet[Color]]= frozenset(("R","G","B","Y","W"))


    def __init__(self, col: CardColor, rank: Optional[CardRank]):
        object.__setattr__(self, "color", col)
        object.__setattr__(self, "rank", -1 if rank is None else rank)

    @classmethod
    def make(cls, c: Card) -> "CardData":
        return cls(c["color"],c["rank"])

    @classmethod
    def makelist(cls, clist: Iterable[Card]) -> List["CardData"]:
        return [cls.make(c) for c in clist]

    @classmethod
    @functools.lru_cache(1)
    def UNKNOWN_CARD(cls) -> "CardData":
        return cls(None, -1)

    @classmethod
    @functools.lru_cache(1)
    def all_valid_cards(cls) -> FrozenSet["CardData"]:
        return frozenset(
            cls(c, r)
            for c in cls.VALID_COLOR
            for r in cls.VALID_RANKS
        )

    @property
    def iter_higher(self) -> Iterator["CardData"]:
        "iterate through all of the cards higher than this card in same colour"
        return (CardData(self.color, rnk) for rnk in range(self.rank +1, 5))

    @functools.cached_property
    def is_known(self) -> bool:
        "returns true if all the data on this card is known"
        return self.color is not None and self.rank != -1

    @functools.cached_property
    def is_unknown(self) -> bool:
        "returns true if everything about this card is unknown"
        return self.color is None and self.rank == -1

    def match_potential_other_known(self, crd: "CardData") -> bool:
        assert crd.is_known
        if self.is_known:
            return self == crd
        else:
            if self.color:
                return self.color == crd.color
            return self.rank == -1 or self.rank == crd.rank

    def filter_card_list(self, other_cards: Iterable["CardData"]) -> Iterable["CardData"]:
        if self.is_known:
            if self in other_cards:
                return [self]
            warnings.warn(f"card {self} not in given other card list\n{other_cards}")
            return []
        elif self.is_unknown:
            return other_cards
        elif self.color is None:
            return [crd for crd in other_cards if self.rank == crd.rank]
        return [crd for crd in other_cards if self.color == crd.color]

    def potential_matches(self, other_cards: Iterable["CardData"]) -> Iterator["CardData"]:
        if self.is_known:
            return (crd for crd in other_cards if crd == self)
        return (crd for crd in other_cards if self.match_potential_other_known(crd))





def card_to_dc(c: Card) -> CardData:
    return CardData(c["color"], c["rank"])

def cardlist_to_dc(clist: Iterable[Card]) -> List[CardData]:
    return [CardData.make(c) for c in clist]

TCard = TypeVar("TCard", bound=Card)

class FireworksDict(TypedDict):
    B: int
    G: int
    R: int
    W: int
    Y: int

FRank = Literal[0,1,2,3,4,5]

@dataclass(init=True, repr=True, eq=True, frozen=True)
class PlayableUselessFuture:
    playable: FrozenSet[CardData]
    useless:FrozenSet[CardData]
    future: FrozenSet[CardData]

    @classmethod
    def make(cls, playable: Set[CardData], useless: Set[CardData], future: Set[CardData]) -> "PlayableUselessFuture":
        return cls(
            frozenset(playable), frozenset(useless), frozenset(future)
        )

    @property
    def get_max_playable_rank(self) -> FRank:
        if not self.playable:
            return 5
        return max(self.playable, key=lambda c: c.rank).rank

@dataclass(init=True, repr=True, eq=True, frozen=True)
class FireworksData:
    B: FRank
    G: FRank
    R: FRank
    W: FRank
    Y: FRank

    def __str__(self):
        return f"fireworks- B:{self.B}, G:{self.G}, R:{self.R}, W:{self.W}, Y:{self.Y}"

    @property
    def tupled(self) -> Tuple[FRank, FRank, FRank, FRank, FRank]:
        return dataclasses.astuple(self)

    @functools.cached_property
    def dicted(self) -> Dict[Color, FRank]:
        return {
            "B": self.B,
            "G": self.G,
            "R": self.R,
            "W": self.W,
            "Y": self.Y
        }

    @functools.cached_property
    def highest_playable_cards(self) -> FrozenSet[CardData]:
        return frozenset(
            CardData(k,v) for k, v in self.dicted.items() if v == self.max_play_rank and v != 5
        )

    @functools.cached_property
    def lowest_playable_cards(self) -> FrozenSet[CardData]:

        return frozenset(
            CardData(k, v) for k, v in self.dicted.items() if v == self.min_play_rank
        )

    @functools.cached_property
    def playable_useless_future(self) -> PlayableUselessFuture:
        playable: Set["CardData"] = set()
        useless: Set["CardData"] = set()
        future: Set["CardData"] = set()
        mydict: Dict[Color, FRank] = self.dicted
        for c in CardData.all_valid_cards():
            if c.rank == mydict[c.color]:
                playable.add(c)
            elif c.rank < mydict[c.color]:
                useless.add(c)
            else:
                future.add(c)
        return PlayableUselessFuture.make(playable, useless, future)

    @property
    def used_cards(self) -> FrozenSet["CardData"]:
        return self.playable_useless_future.useless

    @functools.cached_property
    def min_play_rank(self) -> FRank:
        return min(self.tupled)

    @property
    def max_play_rank(self) -> FRank:
        return self.playable_useless_future.get_max_playable_rank

    @classmethod
    def make(cls, fworks: FireworksDict) -> "FireworksData":
        return cls(fworks["B"],fworks["G"],fworks["R"],fworks["W"],fworks["Y"])





@dataclass(init=True, repr=True, eq=True, frozen=True)
class OtherInfo:
    offset:     int
    hand:       Tuple[CardData]
    playable:   FrozenSet[int]
    unplayable: FrozenSet[int]
    saveable:   FrozenSet[int]
    endangered: FrozenSet[int]
    un_ranks:   FrozenSet[int]
    un_cols:    FrozenSet[int]

    @functools.cached_property
    def full_unknowns(self) -> FrozenSet[int]:
        return self.un_ranks.union(self.un_cols)

    @functools.cached_property
    def oldest_unknown_index(self) -> Optional[int]:
        if self.full_unknowns:
            return min(self.full_unknowns)
        return None

    @property
    def is_oldest_endangered(self) -> bool:
        return self.oldest_unknown_index in self.endangered

    @property
    def is_oldest_saveable(self) -> bool:
        return self.oldest_unknown_index in self.saveable

    @classmethod
    def make(
            cls, offset:int, real: List[CardData], known: List[CardData],
            playable: FrozenSet[CardData], useless: FrozenSet[CardData],
            endangered: FrozenSet[CardData], unique_vis: FrozenSet[CardData]
    ) -> "OtherInfo":
        pl = set()
        ul = set()
        sv = set()
        en = set()
        ur = set()
        uc = set()
        for i, crd in enumerate(known):
            if not crd.is_known:
                if crd.rank == -1:
                    ur.add(i)
                if crd.color is None:
                    uc.add(i)
                crd = real[i]
            if crd in playable:
                pl.add(i)
            elif crd in useless:
                ul.add(i)
            if crd in unique_vis:
                sv.add(i)
                if crd in endangered:
                    en.add(i)
        return OtherInfo(
            offset,
            tuple(real),
            frozenset(pl),
            frozenset(ul),
            frozenset(sv),
            frozenset(en),
            frozenset(ur),
            frozenset(uc)
        )





@dataclass(init=True, repr=True, eq=True, frozen=True)
class MyHandData:
    full_hand: List[CardData]
    possible_cards_in_hand:  Dict[CardData, List[CardData]]
    playable_cards_in_hand:  Dict[CardData, List[CardData]]
    useless_cards_in_hand:   Dict[CardData, List[CardData]]
    future_playable_in_hand: Dict[CardData, List[CardData]]
    endangered_card_in_hand: Dict[CardData, List[CardData]]



    def _prob_calculator(self, candidates: Dict[CardData, List[CardData]]) -> Dict[CardData, float]:
        return dict(
            (c, len(v)/len(self.possible_cards_in_hand[c]))
            for c, v in candidates.items()
        )

    @functools.cached_property
    def probability_hand_card_playable(self) -> Dict[CardData, float]:
        return self._prob_calculator(self.playable_cards_in_hand)
    @functools.cached_property
    def probability_hand_card_useless(self) -> Dict[CardData, float]:
        return self._prob_calculator(self.useless_cards_in_hand)
    @functools.cached_property
    def probability_hand_card_f_playable(self) -> Dict[CardData, float]:
        return self._prob_calculator(self.future_playable_in_hand)
    @functools.cached_property
    def probability_hand_card_endangered(self) -> Dict[CardData, float]:
        return self._prob_calculator(self.endangered_card_in_hand)




    @classmethod
    def make_from_obs(cls, obs: "ObservationData") -> "MyHandData":
        possible_cards_in_hand: Dict[CardData, List[CardData]] = {}
        playable_cards_in_hand: Dict[CardData, List[CardData]] = {}
        useless_cards_in_hand:  Dict[CardData, List[CardData]] = {}
        future_playable_in_hand:Dict[CardData, List[CardData]] = {}
        endangered_card_in_hand:Dict[CardData, List[CardData]] = {}
        for crd in set(obs.my_hand):
            possibles: List[crd] = crd.filter_card_list(obs.unseen)
            possible_cards_in_hand[crd] = possibles
            possible_c: int = len(possibles)
            assert possible_c > 0
            pl = []
            ul = []
            fp = []
            en = []
            for p in possibles:
                if p in obs.playable:
                    pl.append(p)
                elif p in obs.useless:
                    ul.append(p)
                elif p in obs.future_playable:
                    fp.append(p)
                if p in obs.endangered:
                    en.append(p)
            playable_cards_in_hand[crd] = pl
            useless_cards_in_hand[crd]  = ul
            future_playable_in_hand[crd]= fp
            endangered_card_in_hand[crd]= en
        return cls(
            obs.my_hand,
            possible_cards_in_hand,
            playable_cards_in_hand,
            useless_cards_in_hand,
            future_playable_in_hand,
            endangered_card_in_hand
        )

class ObservationDict(TypedDict):
    current_player: int
    current_player_offset: int
    deck_size: int
    discard_pile: List[KnownCard]
    fireworks: FireworksDict
    information_tokens: int
    legal_moves: List[ActionDict]
    life_tokens: int
    card_knowledge: List[List[Union[OwnHand, KnownHand]]]
    observed_hands: List[List[Union[OwnHand, KnownHand]]]
    num_players: int
    vectorized: List[Literal[0, 1]]


@dataclass(init=True, repr=True, eq=True, frozen=True)
class ObservationData:
    "A dataclass wrapper for the observation dict"
    deck_size: int
    "how many cards are still in the deck?"
    unseen: List[CardData]
    "All the cards that we haven't seen"
    discard: List[CardData]
    "all the discarded cards"
    fireworks: FireworksData
    "current state of fireworks"
    infos: int
    "information tokens left"
    legal_moves: List[Action]
    "legal moves that can be performed from this gamestate"
    lives: int
    "lives remaining"
    my_hand: List[CardData]
    "the cards in the player's hand"
    observed: List[List[CardData]]
    "what the player can see in the other hands"
    knowledge: List[List[CardData]]
    "what each player knows about their hands"
    future_playable: FrozenSet[CardData]
    "cards that may be playable in the future"
    playable: FrozenSet[CardData]
    "cards that are currently playable"
    useless: FrozenSet[CardData]
    "cards that can't be played"
    endangered: FrozenSet[CardData]
    "cards that aren't useless but only one instance of them exists"
    one_visible: FrozenSet[CardData]
    "cards that aren't useless but only one instance of them is visible"
    num_players: int
    "how many players are there?"
    _raw: ObservationDict
    "raw observation, may or may not be useful as a fallback"

    @property
    def can_tell(self) -> bool:
        return self.infos > 0

    @functools.cached_property
    def my_hand_dat(self) -> MyHandData:
        "Detailed stats about the player's hand"
        return MyHandData.make_from_obs(self)

    @functools.cached_property
    def others(self) -> Dict[int, OtherInfo]:
        "Detailed stats about other players"
        return dict(
            (i, OtherInfo.make(
                i,
                self.observed[i],
                self.knowledge[i],
                self.playable,
                self.useless,
                self.endangered,
                self.one_visible
            )) for i in range(1, self.num_players)
        )

    @property
    def all_others_visible(self) -> FrozenSet[CardData]:
        return frozenset(
            itertools.chain.from_iterable(
                (crd for crd in self.observed[i] if crd.is_known)
                for i in range(1, self.num_players)
            )
        )

    @staticmethod
    def individual_card_counts() -> Dict[CardData, int]:
        """How many cards are there of each type in the deck?
        Recalculated when called, makes this stored version practically immutable"""
        return dict(
            (crd, 3 if crd.rank == 0 else 1 if crd.rank == 4 else 2) for crd in CardData.all_valid_cards()
        )

    full_deck: ClassVar[Tuple[CardData]]
    "A full deck, as a classvar, for ease of use"

    def mycard_argmax(self, dat: Dict[CardData, float], threshold: float = -math.inf) -> Optional[int]:
        curmax: float = -math.inf
        max_ind: Optional[int] = None
        for ind, crd in enumerate(self.my_hand):
            if dat[crd] >= threshold and dat[crd] > curmax:
                curmax = dat[crd]
                max_ind = ind
        return max_ind

    def mycard_argmin(self, dat: Dict[CardData, float], threshold: float = math.inf) -> Optional[int]:
        curmin: float = math.inf
        min_ind: Optional[int] = None
        for ind, crd in enumerate(self.my_hand):
            if dat[crd] <= threshold and dat[crd] < curmin:
                curmin = dat[crd]
                min_ind = ind
        return min_ind



    @classmethod
    def all_unseen_useless_endangered_onevisible(
            cls, discard: List[CardData], all_visible: List[CardData], fireworks: FireworksData
    ) -> Tuple[List[CardData], Set[CardData], Set[CardData], Set[CardData]]:
        """
        Returns the list of unseen (instances of) cards, useless cards, 'endangered' cards,
        and 'only one visible' cards.
        :param discard: the discard pile
        :param all_visible: all the known cards that we can see
        :param fireworks: data about the fireworks
        :return: a tuple containing:
        <html><ul>
            <li>All unseen instances of cards</li>
            <li>All useless cards</li>
            <li>All visible cards</li>
            <li>All endangered useful cards</li>
            <li>All useful cards where only one instance of them is visible</li>
        </ul></html>
        """

        all_counts: Dict[CardData, int] = cls.individual_card_counts()

        for d in discard:
            all_counts[d] -= 1

        unusable: Set[CardData] = set()
        for crd, c in all_counts.items():
            if crd in unusable:
                continue
            if c == 0:
                unusable.add(crd)
                unusable.union(crd.iter_higher)

        for crd in fireworks.used_cards:
            all_counts[crd] -= 1
            unusable.add(crd)

        seen_counts: Dict[CardData, int] = defaultdict(int)
        for crd in all_visible:
            if crd.is_known and crd not in unusable:
                seen_counts[crd] += 1
                all_counts[crd] -= 1

        unseen: List[CardData] = []
        endangered: Set[CardData] = set()

        for crd, c in all_counts.items():
            if c > 0:
                if c == 1:
                    endangered.add(crd)
                unseen.append(crd)

        return unseen, unusable, endangered, set(crd for crd, q in seen_counts.items() if q == 1)

    @classmethod
    def make(cls, obs: ObservationDict) -> "ObservationData":
        "Creates this, from the dictionary form of the observations"

        fireworks: FireworksData = FireworksData.make(obs["fireworks"])

        observed: List[List[CardData]] = []
        knowledge:List[List[CardData]] = []
        visible:  List[CardData] = []
        my_hand: List[CardData] = []

        for p in range(obs["num_players"]):
            knowledge.append([CardData.make(c) for c in obs["card_knowledge"][p]])
            current_obs: List[CardData] = [CardData.make(c) for c in obs["observed_hands"][p]]
            observed.append(current_obs)
            if p == 0:
                my_hand = current_obs
                visible += [crd for crd in my_hand if crd.is_known]
                continue
            visible += current_obs

        assert any(my_hand)
        #hand_size = len(my_hand)

        discard: List[CardData] = CardData.makelist(obs["discard_pile"])

        #play_useless_future: PlayableUselessFuture = fireworks.playable_useless_future

        unseen: List[CardData]
        useless: Set[CardData]
        endangered: Set[CardData]
        one_visible:Set[CardData]
        unseen, useless, endangered, one_visible = cls.all_unseen_useless_endangered_onevisible(
            discard,
            visible,
            fireworks
        )
        future_playable: FrozenSet[CardData] = frozenset(fireworks.playable_useless_future.future - useless)
        playable: FrozenSet[CardData] = frozenset(fireworks.playable_useless_future.playable - useless)
        useless: FrozenSet[CardData] = frozenset(useless.union(fireworks.playable_useless_future.useless))
        endangered: FrozenSet[CardData] = frozenset(endangered.difference(useless))
        one_visible: FrozenSet[CardData] = frozenset(one_visible.difference(useless))

        return cls(
            obs["deck_size"],
            unseen,
            discard,
            fireworks,
            obs["information_tokens"],
            obs["legal_moves"],
            obs["life_tokens"],
            my_hand,
            observed,
            knowledge,
            future_playable,
            playable,
            useless,
            endangered,
            one_visible,
            obs["num_players"],
            obs
        )

ObservationData.full_deck = tuple(
    itertools.chain.from_iterable(
        [crd] * q for crd, q
        in ObservationData.individual_card_counts().items()
    )
)

#%%

T = TypeVar("T")

def get_any(it: Iterable[T]) -> Optional[T]:
    """
    Used to return an arbitrary item from an iterable.
    :param it: the iterable we want an arbitrary item from
    :return: the first item from that iterable, or null if it's empty
    """
    an_iter: Iterator[T] = it.__iter__()
    try:
        return an_iter.__next__()
    except StopIteration:
        return None

def get_first_where(it: Iterable[T], where: Callable[[T],bool]) -> Optional[T]:
    """
    Attempts to return first item from iterable meeting a condition
    :param it: the iterable we want the first matching item from
    :param where: the condition we want this item to satisfy
    :return: the first item from that iterable matching that condition, or None if it's empty/none match
    """
    return get_any(filter(where, it))


def get_ind_first_where(it: Iterable[T], where: Callable[[T], bool]) -> Optional[int]:
    """
    Attempts to return the index of first item from iterable meeting a condition,
    as indexed by enumerate
    :param it: the iterable we want the first matching item from
    :param where: the condition we want this item to satisfy
    :return: the enumerated index of first item from that iterable matching that condition,
    or None if it's empty/none match
    """
    if (res := get_first_where(enumerate(it), lambda kv: where(kv[1]))) is not None:
        return res[1]
    return None



#%%

class RuleType(IntFlag):
    PLAY = auto()
    TELL = auto()
    DISCARD = auto()
    TELL_FIVES = auto()
    TELL_PLAYABLE = auto()
    TELL_USEFUL_ONES = auto()
    TELL_USEFUL_TWOS = auto()
    TELL_HIGHEST_PLAYABLE = auto()
    TELL_LOWEST_PLAYABLE = auto()
    TELL_NEXT = auto()
    TELL_USELESS = auto()
    TELL_SAVEABLE = auto()
    TELL_ENDANGERED = auto()

class RulesEnum(Enum):

    def __new__(
            cls,
            rule_type: RuleType,
            param: Optional[float] = None

    ) -> "RulesEnum":
        """
        Attempts to create the individual members of this enum.
        :param val:
        :param arity: How many trainable arguments does this rule have?
        :param filter: Are there any special conditions that need
        to be met for this rule to be applied? If None or idc, it's always applicable.
        """
        val: int = len(cls.__members__)
        r: "RulesEnum" = object.__new__(cls)
        r._value_ = val
        r.rule_type = rule_type
        r._param = param if param is not None else 0.0
        return r

    #PLAY_MOST_PLAYABLE_CARD = (RuleType.PLAY, 0)
    PLAY_MOST_PLAYABLE_CARD_90 = (RuleType.PLAY, 0.9)
    PLAY_MOST_PLAYABLE_CARD_80 = (RuleType.PLAY, 0.8)
    PLAY_MOST_PLAYABLE_CARD_60 = (RuleType.PLAY, 0.6)
    PLAY_MOST_PLAYABLE_CARD_50 = (RuleType.PLAY, 0.5)
    PLAY_MOST_PLAYABLE_CARD_40 = (RuleType.PLAY, 0.4)
    PLAY_MOST_DEFINITELY_PLAYABLE_CARD = (RuleType.PLAY, 1)

    DISCARD_OLDEST_UNPLAYABLE_CARD = RuleType.DISCARD
    DISCARD_OLDEST_UNKNOWN_CARD = RuleType.DISCARD

    TELL_PLAYER_ABOUT_OLDEST_UNKNOWN_CARD_IF_NOT_USELESS_RANK = RuleType.TELL
    TELL_PLAYER_ABOUT_OLDEST_UNKNOWN_CARD_IF_NOT_USELESS_COLOR = RuleType.TELL

    TELL_PLAYER_ABOUT_FIVES_RANK = RuleType.TELL | RuleType.TELL_FIVES
    TELL_PLAYER_ABOUT_FIVES_COLOR = RuleType.TELL | RuleType.TELL_FIVES

    TELL_PLAYER_ABOUT_PLAYABLE_CARD_RANK = RuleType.TELL | RuleType.TELL_PLAYABLE
    TELL_PLAYER_ABOUT_PLAYABLE_CARD_COLOR = RuleType.TELL | RuleType.TELL_PLAYABLE


    TELL_PLAYER_ABOUT_PLAYABLE_ONES_RANK =  RuleType.TELL | RuleType.TELL_PLAYABLE | RuleType.TELL_USEFUL_ONES
    TELL_PLAYER_ABOUT_PLAYABLE_ONES_COLOR =  RuleType.TELL | RuleType.TELL_PLAYABLE | RuleType.TELL_USEFUL_ONES
    TELL_PLAYER_ABOUT_USEFUL_TWOS_RANK =  RuleType.TELL | RuleType.TELL_SAVEABLE | RuleType.TELL_USEFUL_TWOS
    TELL_PLAYER_ABOUT_USEFUL_TWOS_COLOR =  RuleType.TELL | RuleType.TELL_SAVEABLE | RuleType.TELL_USEFUL_TWOS

    TELL_NEXT_PLAYER_ABOUT_PLAYABLE_CARD_RANK = RuleType.TELL | RuleType.TELL_PLAYABLE | RuleType.TELL_NEXT
    TELL_NEXT_PLAYER_ABOUT_PLAYABLE_CARD_COLOR = RuleType.TELL | RuleType.TELL_PLAYABLE | RuleType.TELL_NEXT
    TELL_NEXT_PLAYER_ABOUT_LOWEST_PLAYABLE_COLOR = RuleType.TELL | RuleType.TELL_PLAYABLE | RuleType.TELL_NEXT |  RuleType.TELL_LOWEST_PLAYABLE
    TELL_NEXT_PLAYER_ABOUT_LOWEST_PLAYABLE_RANK = RuleType.TELL | RuleType.TELL_PLAYABLE | RuleType.TELL_NEXT | RuleType.TELL_LOWEST_PLAYABLE


    TELL_PLAYER_ABOUT_HIGHEST_PLAYABLE_COLOR = RuleType.TELL | RuleType.TELL_PLAYABLE | RuleType.TELL_HIGHEST_PLAYABLE
    TELL_PLAYER_ABOUT_HIGHEST_PLAYABLE_RANK = RuleType.TELL | RuleType.TELL_PLAYABLE | RuleType.TELL_HIGHEST_PLAYABLE

    TELL_PLAYER_ABOUT_LOWEST_PLAYABLE_COLOR = RuleType.TELL | RuleType.TELL_PLAYABLE | RuleType.TELL_LOWEST_PLAYABLE
    TELL_PLAYER_ABOUT_LOWEST_PLAYABLE_RANK = RuleType.TELL | RuleType.TELL_PLAYABLE | RuleType.TELL_LOWEST_PLAYABLE


    TELL_NEXT_PLAYER_ABOUT_UNPLAYABLE_REMAINING_INFO = RuleType.TELL | RuleType.TELL_NEXT | RuleType.TELL_USELESS
    TELL_NEXT_PLAYER_ABOUT_UNPLAYABLE_RANK = RuleType.TELL | RuleType.TELL_NEXT | RuleType.TELL_USELESS
    TELL_NEXT_PLAYER_ABOUT_UNPLAYABLE_COLOR = RuleType.TELL | RuleType.TELL_NEXT | RuleType.TELL_USELESS

    TELL_PLAYER_ABOUT_UNKNOWN_REMAINING_INFO = RuleType.TELL | RuleType.TELL_USELESS
    TELL_PLAYER_ABOUT_USELESS_RANK = RuleType.TELL | RuleType.TELL_USELESS
    TELL_PLAYER_ABOUT_USELESS_COLOR = RuleType.TELL | RuleType.TELL_USELESS
    TELL_PLAYER_ABOUT_RANKS_BELOW_MIN_USEFUL = RuleType.TELL | RuleType.TELL_USELESS

    TELL_PLAYER_ABOUT_OLDEST_UNKNOWN_CARD_IF_UNIQUE_VISIBLE_RANK = RuleType.TELL | RuleType.TELL_SAVEABLE
    TELL_PLAYER_ABOUT_OLDEST_UNKNOWN_CARD_IF_UNIQUE_VISIBLE_COLOR = RuleType.TELL | RuleType.TELL_SAVEABLE
    TELL_PLAYER_ABOUT_OLDEST_UNKNOWN_CARD_IF_ENDANGERED_RANK = RuleType.TELL | RuleType.TELL_ENDANGERED
    TELL_PLAYER_ABOUT_OLDEST_UNKNOWN_CARD_IF_ENDANGERED_COLOR = RuleType.TELL |  RuleType.TELL_ENDANGERED

    TELL_NEXT_PLAYER_ABOUT_OLDEST_UNKNOWN_CARD_IF_UNIQUE_VISIBLE_RANK = RuleType.TELL | RuleType.TELL_NEXT | RuleType.TELL_SAVEABLE
    TELL_NEXT_PLAYER_ABOUT_OLDEST_UNKNOWN_CARD_IF_UNIQUE_VISIBLE_COLOR = RuleType.TELL | RuleType.TELL_NEXT | RuleType.TELL_SAVEABLE
    TELL_NEXT_PLAYER_ABOUT_OLDEST_UNKNOWN_CARD_IF_ENDANGERED_RANK = RuleType.TELL | RuleType.TELL_NEXT | RuleType.TELL_ENDANGERED
    TELL_NEXT_PLAYER_ABOUT_OLDEST_UNKNOWN_CARD_IF_ENDANGERED_COLOR = RuleType.TELL | RuleType.TELL_NEXT | RuleType.TELL_ENDANGERED

    @classmethod
    def to_tuple(cls) -> Tuple["RulesEnum"]:
        return tuple(cls.__members__.values())

class NoRulesLeftException(Exception):
    pass



@dataclass(init=True, repr=True, eq=True, frozen=True)
class ChromosomeInfo:
    """
    A dictionary of information which will be given to each chromosome when
    they're being asked to provide info to the agent. This may or may not be
    used when a chromosome decides which list of rules to give to the agent.
    """
    can_discard: bool
    "Is the player able to discard (not at max info tokens)?"
    info: int
    "How many info tokens are left?"
    lives: int
    "How many lives are left?"
    fireworks: FireworksData
    "What fireworks have been played?"
    deck_left: int
    "How much of the deck is left?"
    player_index: int
    "Which index player is this?"
    has_discarded: bool
    "Have any cards been discarded yet?"

    @classmethod
    def make(cls, obs: ObservationData, player_index: int, max_info: int = 8):
        """
        easy way of making this
        :param obs: observation data
        :param player_index: which player is this?
        :param max_info: how many info tokens are left?
        :return: a ChromosomeInfo representing this data.
        """
        return cls(
            obs.infos > max_info,
            obs.infos,
            obs.lives,
            obs.fireworks,
            obs.deck_size,
            player_index,
            len(obs.discard) > 0
        )


class I_AgentChromosome(abc.ABC):
    """
    An interface for the chromosomes, exposing only the method
    which the agent calls in order to obtain the necessary
    ordered rules it needs for purposes of playing the game.
    """

    @abc.abstractmethod
    def get_rules(self, info: ChromosomeInfo) -> Iterator[RulesEnum]:
        """
        Given a ChromosomeInfo object, return the appropriate ordered list of rules
        for the situation
        :param info: The info to give to the chromosome, for purposes of choosing
        what ruleset to return
        :return: an iterator for the appropriate ruleset.
        """
        pass

    @staticmethod
    def get_ordered_rules_iterator(
            order: np.ndarray, rules: Tuple[RulesEnum] = RulesEnum.to_tuple()
    ) -> Iterator[RulesEnum]:

        #cls.rules_list.copy()
        #srt: List[RulesEnum] = [rlist[ind] for ind in argsorted]

        return (rules[ind] for ind in np.argsort(np.subtract(0, order), axis=None))


class default_chromo(I_AgentChromosome):
    def __init__(self):
        pass

    def get_rules(self, info: ChromosomeInfo) -> Tuple[RulesEnum]:
        return self.get_ordered_rules_iterator(np.asarray([
            0.68370241, 0.83874948, 0.5478453 , 0.68230499 ,0.93287692, 0.65374768,
            0.69221556 ,0.63221866, 0.53491521, 0.86149117 ,0.70583545, 0.63037599,
            0.65968975 ,0.63033461, 0.63542557, 0.47767838, 0.49073739 ,0.7273925,
            0.53746922 ,0.41781634 ,0.60722811, 0.60984701, 0.50664009, 0.72302587,
            0.69323435 ,0.58906901, 0.5093967 , 0.35338183, 0.7197041 , 0.71586255,
            0.84727529 ,0.61581697, 0.37011163 ,0.75167379, 0.86199249 ,0.53208804,
            0.7028286 , 0.74466516 ,0.78216393, 0.6242901 , 0.62842792]))

#%%






class MyAgent(Agent):
    """Agent that applies a simple heuristic."""


    def __init__(self, config, chromosome: I_AgentChromosome=None,  *args, **kwargs):
        """Initialize the agent."""
        self.config = config
        if chromosome is None:
            self.chromosome: I_AgentChromosome = default_chromo()
        else:
            self.chromosome: I_AgentChromosome = chromosome
        assert issubclass(self.chromosome.__class__, I_AgentChromosome)

        self.not_played: bool = True
        self.player_num: int = 0


        # Extract max info tokens or set default to 8.
        self.max_information_tokens = config.get('information_tokens', 8)

    def reset(self, config):
        self.config = config
        self.max_information_tokens = config.get('information_tokens', 8)
        self.player_num = 0
        self.not_played = True


    def act(self, observation: ObservationDict) -> Union[ActionDict, None]:
        # this function is called for every player on every turn
        """Act based on an observation."""

        if observation['current_player_offset'] != 0:
            # but only the player with offset 0 is allowed to make an action.  The other players are just observing.
            if self.not_played:
                self.player_num = (self.player_num + 1) % 4
            return None

        if self.not_played:
            self.not_played = False

        observed: ObservationData = ObservationData.make(observation)

        try:
            for act_r in self.pick_action(observed):
                if (act := act_r[0]) in observed.legal_moves:
                    return act
        except AssertionError:
            print("oh dear.")
        except NoRulesLeftException:
            if observed.infos < self.max_information_tokens and (ind := get_ind_first_where(observed.my_hand, lambda crd: crd.is_unknown)) is not None:
                if (act := actdiscard(ind)) in observed.legal_moves:
                    return act
            if (maxp := observed.mycard_argmax(
                observed.my_hand_dat.probability_hand_card_playable,
                -5
            )) is not None:
                if (act := actplay(maxp)) in observed.legal_moves:
                    return act
            return observed.legal_moves[0]


    def pick_action(self, observed: ObservationData) -> Iterator[Tuple[Action, RulesEnum]]:

        not_tried_next: bool = True

        filter_these_out: Set[RuleType] = set()
        if observed.can_tell:
            if observed.infos == self.max_information_tokens:
                filter_these_out.add(RuleType.DISCARD)
            all_vis = observed.all_others_visible
            if observed.useless.isdisjoint(all_vis):
                filter_these_out.add(RuleType.TELL_USELESS)
            if any(crd.rank == 4 for crd in all_vis):
                filter_these_out.add(RuleType.TELL_FIVES)
            if (min_playable := observed.fireworks.min_play_rank) > 0:
                filter_these_out.add(RuleType.TELL_USEFUL_ONES)
                if min_playable > 1:
                    filter_these_out.add(RuleType.TELL_USEFUL_TWOS)
            if observed.playable.isdisjoint(all_vis):
                filter_these_out.add(RuleType.TELL_PLAYABLE)
            else:
                if all_vis.isdisjoint(observed.fireworks.lowest_playable_cards):
                    filter_these_out.add(RuleType.TELL_LOWEST_PLAYABLE)
                if all_vis.isdisjoint(observed.fireworks.highest_playable_cards):
                    filter_these_out.add(RuleType.TELL_HIGHEST_PLAYABLE)
            if not observed.one_visible:
                filter_these_out.add(RuleType.TELL_SAVEABLE)
            if not observed.endangered:
                filter_these_out.add(RuleType.TELL_ENDANGERED)
        else:
            filter_these_out.add(RuleType.TELL)


        for rule in self.chromosome.get_rules(ChromosomeInfo.make(
            observed,
            self.player_num,
            self.max_information_tokens
        )):

            if any(rule.rule_type & f for f in filter_these_out):
                continue

            if rule.rule_type == RuleType.PLAY:
                if (maxp := observed.mycard_argmax(
                        observed.my_hand_dat.probability_hand_card_playable,
                    rule._param)
                ) is not None:
                    yield actplay(maxp), rule
                continue
            elif rule.rule_type == RuleType.DISCARD:
                if rule == RulesEnum.DISCARD_OLDEST_UNPLAYABLE_CARD:
                    if (ind_useless:= get_first_where(
                            enumerate(observed.my_hand), lambda kv: kv[1] in observed.useless)
                    ) is not None:
                        yield actdiscard(ind_useless[0]), rule
                elif rule == RulesEnum.DISCARD_OLDEST_UNKNOWN_CARD:
                    if (ind_unknown := get_first_where(
                            enumerate(observed.my_hand), lambda kv: kv[1].is_unknown)
                    ) is not None:
                        yield actdiscard(ind_unknown[0]), rule
                else:
                    raise AssertionError(f"Unexpected rule in 'discard' branch!\n{rule}")
                continue
            elif rule.rule_type & RuleType.TELL:
                if rule == RulesEnum.TELL_PLAYER_ABOUT_OLDEST_UNKNOWN_CARD_IF_NOT_USELESS_RANK:
                    for p in observed.others.values():
                        if p.oldest_unknown_index not in p.unplayable:
                            yield actrank(p, p.oldest_unknown_index), rule
                            break
                elif rule == RulesEnum.TELL_PLAYER_ABOUT_OLDEST_UNKNOWN_CARD_IF_NOT_USELESS_COLOR:
                    for p in observed.others.values():
                        if p.oldest_unknown_index not in p.unplayable:
                            yield actcolor(p, p.oldest_unknown_index), rule
                            break
                elif rule.rule_type & RuleType.TELL_FIVES:
                    if rule == RulesEnum.TELL_PLAYER_ABOUT_FIVES_RANK:
                        for p in observed.others.values():
                            if any(p.hand[crd].rank == 4 for crd in p.un_ranks):
                                yield actrank(p, 4), rule
                                break
                    elif rule == RulesEnum.TELL_PLAYER_ABOUT_FIVES_COLOR:
                        for p in observed.others.values():
                            if (ind := get_any(p.hand[crd] == 4 for crd in p.un_cols)) is not None:
                                yield actcolor(p, p.hand[ind].color), rule
                                break
                    else:
                        raise AssertionError(f"Unexpected rule in 'fives' branch!\n{rule}")
                    continue
                elif rule.rule_type & RuleType.TELL_USEFUL_ONES:
                    if rule == RulesEnum.TELL_PLAYER_ABOUT_PLAYABLE_ONES_RANK:
                        for p in observed.others.values():
                            if any(p.hand[crd].rank == 0 for crd in p.un_ranks):
                                yield actrank(p, 0), rule
                                break
                    elif rule == RulesEnum.TELL_PLAYER_ABOUT_PLAYABLE_ONES_COLOR:
                        for p in observed.others.values():
                            if (ind := get_any(p.hand[crd] == 0 for crd in p.un_cols)) is not None:
                                yield actcolor(p, p.hand[ind].color), rule
                                break
                    else:
                        raise AssertionError(f"Unexpected rule in 'ones' branch!\n{rule}")
                    continue
                elif rule.rule_type & RuleType.TELL_USEFUL_TWOS:
                    if rule == RulesEnum.TELL_PLAYER_ABOUT_USEFUL_TWOS_RANK:
                        for p in observed.others.values():
                            if any(p.hand[crd].rank == 1 for crd in p.un_ranks.union(p.saveable)):
                                yield actrank(p, 1), rule
                                break
                    elif rule == RulesEnum.TELL_PLAYER_ABOUT_USEFUL_TWOS_COLOR:
                        for p in observed.others.values():
                            if (ind := get_any(p.hand[crd] == 1 for crd in p.un_cols.union(p.saveable))) is not None:
                                yield actcolor(p, p.hand[ind].color), rule
                                break
                    else:
                        raise AssertionError(f"Unexpected rule in 'ones' branch!\n{rule}")
                    continue
                elif rule.rule_type & RuleType.TELL_NEXT:
                    nextp: OtherInfo = observed.others[1]
                    if not_tried_next:
                        not_tried_next = True
                        if not nextp.un_cols.union(nextp.un_ranks):
                            filter_these_out.add(RuleType.TELL_NEXT)
                            continue
                    if rule.rule_type & RuleType.TELL_PLAYABLE:
                        if rule == RulesEnum.TELL_NEXT_PLAYER_ABOUT_PLAYABLE_CARD_RANK:
                            if (ind := get_any(nextp.un_ranks.intersection(nextp.playable))) is not None:
                                yield actrank(nextp, nextp.hand[ind].rank), rule
                                break
                        elif rule == RulesEnum.TELL_NEXT_PLAYER_ABOUT_PLAYABLE_CARD_COLOR:
                            if (ind := get_any(nextp.un_cols.intersection(nextp.playable))) is not None:
                                yield actrank(nextp, nextp.hand[ind].color), rule
                                break
                        elif rule.rule_type & RuleType.TELL_HIGHEST_PLAYABLE:
                            if rule == RulesEnum.TELL_NEXT_PLAYER_ABOUT_HIGHEST_PLAYABLE_COLOR:
                                if nextp.un_cols and (tell_ind := max(
                                        nextp.un_cols, key=lambda k: nextp.hand[k].rank)
                                ) is not None:
                                    yield actcolor(nextp, nextp.hand[tell_ind].color), rule
                            elif rule == RulesEnum.TELL_NEXT_PLAYER_ABOUT_HIGHEST_PLAYABLE_RANK:
                                if nextp.un_ranks and (tell_ind := max(
                                        nextp.un_ranks, key=lambda k: nextp.hand[k].rank)
                                ) is not None:
                                    yield actrank(nextp, nextp.hand[tell_ind].rank), rule
                            else:
                                print(f"unexpected {rule}!")
                                assert False
                            continue
                        elif rule.rule_type & RuleType.TELL_LOWEST_PLAYABLE:
                            if rule == RulesEnum.TELL_NEXT_PLAYER_ABOUT_LOWEST_PLAYABLE_COLOR:
                                if nextp.un_cols and (tell_ind := min(
                                        nextp.un_cols, key=lambda k: nextp.hand[k].rank)
                                ) is not None:
                                    yield actcolor(nextp, nextp.hand[tell_ind].color), rule
                            elif rule == RulesEnum.TELL_NEXT_PLAYER_ABOUT_LOWEST_PLAYABLE_RANK:
                                if nextp.un_ranks and (tell_ind := min(
                                        nextp.un_ranks, key=lambda k: nextp.hand[k].rank)
                                ) is not None:
                                    yield actrank(nextp, nextp.hand[tell_ind].rank), rule
                            else:
                                print(f"unexpected {rule}!")
                                assert False
                            continue
                        else:
                            print(f"unexpected {rule}!")
                            assert False
                    elif rule.rule_type & RuleType.TELL_USELESS:
                        if not nextp.unplayable:
                            continue
                        elif rule == RulesEnum.TELL_NEXT_PLAYER_ABOUT_UNPLAYABLE_REMAINING_INFO:
                            if (ind := get_any((nextp.un_cols ^ nextp.un_ranks).union(nextp.unplayable))) is not None:
                                if ind in nextp.un_cols:
                                    yield actcolor(nextp, nextp.hand[ind].color), rule
                                else:
                                    yield actrank(nextp, nextp.hand[ind].rank), rule
                            continue
                        elif rule == RulesEnum.TELL_NEXT_PLAYER_ABOUT_UNPLAYABLE_COLOR:
                            if (ind := get_any(nextp.un_cols.union(nextp.unplayable))) is not None:
                                yield actcolor(nextp, nextp.hand[ind].color), rule
                            continue
                        elif rule == RulesEnum.TELL_NEXT_PLAYER_ABOUT_UNPLAYABLE_RANK:
                            if (ind := get_any(nextp.un_ranks.union(nextp.unplayable))) is not None:
                                yield actrank(nextp, nextp.hand[ind].rank), rule
                            continue
                        else:
                            print(f"unexpected {rule}!")
                            assert False
                        continue
                    elif rule.rule_type & RuleType.TELL_SAVEABLE:
                        if nextp.oldest_unknown_index is None or nextp.oldest_unknown_index not in nextp.saveable:
                            continue
                        elif rule == RulesEnum.TELL_NEXT_PLAYER_ABOUT_OLDEST_UNKNOWN_CARD_IF_UNIQUE_VISIBLE_COLOR:
                            yield actcolor(nextp, nextp.hand[nextp.oldest_unknown_index].color), rule
                        elif rule == RulesEnum.TELL_NEXT_PLAYER_ABOUT_OLDEST_UNKNOWN_CARD_IF_UNIQUE_VISIBLE_RANK:
                            yield actrank(nextp, nextp.hand[nextp.oldest_unknown_index].rank), rule
                        else:
                            print(f"unexpected {rule}!")
                            assert False
                        continue
                    elif rule.rule_type & RuleType.TELL_ENDANGERED:
                        if nextp.oldest_unknown_index is None or nextp.oldest_unknown_index not in nextp.endangered:
                            continue
                        elif rule == RulesEnum.TELL_NEXT_PLAYER_ABOUT_OLDEST_UNKNOWN_CARD_IF_ENDANGERED_COLOR:
                            yield actcolor(nextp, nextp.hand[nextp.oldest_unknown_index].color), rule
                        elif rule == RulesEnum.TELL_NEXT_PLAYER_ABOUT_OLDEST_UNKNOWN_CARD_IF_ENDANGERED_RANK:
                            yield actrank(nextp, nextp.hand[nextp.oldest_unknown_index].rank), rule
                        else:
                            print(f"unexpected {rule}!")
                            assert False
                        continue
                    else:
                        print(f"unexpected {rule}!")
                        assert False
                    continue
                elif rule.rule_type & RuleType.TELL_PLAYABLE:
                    if rule == RulesEnum.TELL_PLAYER_ABOUT_PLAYABLE_CARD_RANK:
                        for p in observed.others.values():
                            if (ind := get_any(p.un_ranks.intersection(p.playable))) is not None:
                                yield actrank(p, p.hand[ind].rank), rule
                                break
                    elif rule == RulesEnum.TELL_PLAYER_ABOUT_PLAYABLE_CARD_COLOR:
                        for p in observed.others.values():
                            if (ind := get_any(p.un_cols.intersection(p.playable))) is not None:
                                yield actrank(p, p.hand[ind].color), rule
                                break
                    elif rule.rule_type & RuleType.TELL_HIGHEST_PLAYABLE:
                        if rule == RulesEnum.TELL_PLAYER_ABOUT_HIGHEST_PLAYABLE_COLOR:
                            for p in observed.others.values():
                                if p.un_cols and (tell_ind := max(
                                        p.un_cols, key=lambda k: p.hand[k].rank)
                                ) is not None:
                                    yield actcolor(p, p.hand[tell_ind].color), rule
                                    break
                        elif rule == RulesEnum.TELL_PLAYER_ABOUT_HIGHEST_PLAYABLE_RANK:
                            for p in observed.others.values():
                                if p.un_ranks and (tell_ind := max(
                                        p.un_ranks, key=lambda k: p.hand[k].rank)
                                ) is not None:
                                    yield actrank(p, p.hand[tell_ind].rank), rule
                                    break
                        else:
                            print(f"unexpected {rule}!")
                            assert False
                        continue
                    elif rule.rule_type & RuleType.TELL_LOWEST_PLAYABLE:
                        if rule == RulesEnum.TELL_PLAYER_ABOUT_LOWEST_PLAYABLE_COLOR:
                            for p in observed.others.values():
                                if p.un_cols and (tell_ind := min(
                                        p.un_cols, key=lambda k: p.hand[k].rank)
                                ) is not None:
                                    yield actcolor(p, p.hand[tell_ind].color), rule
                                break
                        elif rule == RulesEnum.TELL_PLAYER_ABOUT_LOWEST_PLAYABLE_RANK:
                            for p in observed.others.values():
                                if p.un_ranks and (tell_ind := min(
                                        p.un_ranks, key=lambda k: p.hand[k].rank)
                                ) is not None:
                                    yield actrank(p, p.hand[tell_ind].rank), rule
                        else:
                            print(f"unexpected {rule}!")
                            assert False
                        continue

                    else:
                        print(f"unexpected {rule}!")
                        assert False
                    continue
                elif rule.rule_type & RuleType.TELL_USELESS:
                    if rule == RulesEnum.TELL_PLAYER_ABOUT_RANKS_BELOW_MIN_USEFUL:
                        if observed.fireworks.min_play_rank == 0:
                            continue
                        for p in observed.others.values():
                            if p.unplayable and (ran := get_first_where(
                                [p.hand[r].rank for r in p.un_ranks],
                                    lambda r: r < observed.fireworks.min_play_rank
                            )) is not None:
                                yield actrank(p, ran), rule
                                break
                    elif rule == RulesEnum.TELL_PLAYER_ABOUT_UNKNOWN_REMAINING_INFO:
                        for p in observed.others.values():
                            if (ind := get_any(p.unplayable.union(p.un_ranks ^ p.un_cols))) is not None:
                                if ind in p.un_ranks:
                                    yield actrank(p, p.hand[ind].rank), rule
                                else:
                                    yield actcolor(p, p.hand[ind].color), rule
                    elif rule == RulesEnum.TELL_PLAYER_ABOUT_USELESS_COLOR:
                        for p in observed.others.values():
                            if p.unplayable and (ind := get_any(
                                [u in p.unplayable for u in p.un_cols],
                            )) is not None:
                                yield actcolor(p, p.hand[ind].color), rule
                                break
                    elif rule == RulesEnum.TELL_PLAYER_ABOUT_USELESS_RANK:
                        for p in observed.others.values():
                            if p.unplayable and (ind := get_any(
                                [u in p.unplayable for u in p.un_ranks],
                            )) is not None:
                                yield actrank(p, p.hand[ind].rank), rule
                                break
                    else:
                        print(f"unexpected {rule}!")
                        assert False
                    continue
                elif rule.rule_type & RuleType.TELL_SAVEABLE:
                    if rule == RulesEnum.TELL_PLAYER_ABOUT_OLDEST_UNKNOWN_CARD_IF_UNIQUE_VISIBLE_RANK:
                        for p in observed.others.values():
                            if p.saveable and p.oldest_unknown_index is not None and p.oldest_unknown_index in p.saveable:
                                yield actrank(p, p.hand[p.oldest_unknown_index].rank), rule
                                break
                    elif rule == RulesEnum.TELL_PLAYER_ABOUT_OLDEST_UNKNOWN_CARD_IF_UNIQUE_VISIBLE_COLOR:
                        for p in observed.others.values():
                            if p.saveable and p.oldest_unknown_index is not None and p.oldest_unknown_index in p.saveable:
                                yield actcolor(p, p.hand[p.oldest_unknown_index].color), rule
                                break
                    else:
                        print(f"unexpected {rule}!")
                        assert False
                    continue
                elif rule.rule_type & RuleType.TELL_ENDANGERED:
                    if rule == RulesEnum.TELL_PLAYER_ABOUT_OLDEST_UNKNOWN_CARD_IF_ENDANGERED_RANK:
                        for p in observed.others.values():
                            if p.endangered and p.oldest_unknown_index is not None and p.oldest_unknown_index in p.endangered:
                                yield actrank(p, p.hand[p.oldest_unknown_index].rank), rule
                                break
                    elif rule == RulesEnum.TELL_PLAYER_ABOUT_OLDEST_UNKNOWN_CARD_IF_ENDANGERED_COLOR:
                        for p in observed.others.values():
                            if p.endangered and p.oldest_unknown_index is not None and p.oldest_unknown_index in p.endangered:
                                yield actcolor(p, p.hand[p.oldest_unknown_index].color), rule
                                break
                    else:
                        print(f"unexpected {rule}!")
                        assert False
                    continue
                else:
                    print(f"unexpected {rule}!")
                    assert False
                continue
            else:
                return AssertionError(f"Completely unexpected rule!\n{rule}")
        raise NoRulesLeftException(f"Out of rules to fire! Rule list:\n"+
                              f"{pprint.pformat(self.chromosome.get_rules(ChromosomeInfo.make(observed,self.player_num,self.max_information_tokens)))}")
