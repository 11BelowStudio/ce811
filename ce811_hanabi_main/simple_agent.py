from __future__ import annotations
"""Simple Agent."""
# Modified from https://github.com/deepmind/hanabi-learning-environment/blob/master/hanabi_learning_environment/agents/simple_agent.py 

from hanabi_learning_environment.rl_env import Agent

from typing import TypedDict, Literal, Union, List

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


class SimpleAgent(Agent):
    """Agent that applies a simple heuristic."""

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent."""
        self.config = config
        # Extract max info tokens or set default to 8.
        self.max_information_tokens: int = config.get('information_tokens', 8)


    def playable_card(self, card: KnownCard, fireworks: FireworksDict) -> bool:
        """A card is playable if it can be placed on the fireworks pile."""
        return card['rank'] == fireworks[card['color']]

    def act(self, observation: ObservationDict) -> Union[ActionDict, None]:
        """Act based on an observation."""


        if observation['current_player_offset'] != 0:
            return None # No action returned (because it's not our turn!)

        fireworks: FireworksDict = observation['fireworks'] # This is a dictionary like {'R': 0, 'Y': 1, 'G': 0, 'W': 1, 'B': 0}


        card_knowledge_about_own_hand: OwnHand = observation['card_knowledge'][0]
        '''
        card_knowledge_about_own_hand is a list of dictionaries, like  
            [{'color': None, 'rank': None}, {'color': None, 'rank': None}, 
            {'color': 'B', 'rank': None}, {'color': None, 'rank': None}, 
            {'color': 'W', 'rank': None}] 
        This particular card_knowledge_about_own_hand list tells us that we have 5 cards
        in our hand, we the only things we know about them is that the 3rd card is blue,
        and the 5th card is white.  We don't know any of the card ranks yet.
        '''

        #print(card_knowledge_about_own_hand)

        #print(observation['card_knowledge'])


        # Check if there are any pending hints and play the card corresponding to
        # the hint.
        for card_index, hint in enumerate(card_knowledge_about_own_hand):
            if hint['color'] is not None and hint['rank'] != -1 and self.playable_card(hint, fireworks):
                # This card has had both hints and we know it's playable, we play it.
                return {'action_type': 'PLAY', 'card_index': card_index}
            elif hint['rank'] == 0 and any((f == hint['rank'] for f in fireworks.values())):
                return {'action_type': 'PLAY', 'card_index': card_index}
            #elif hint['rank'] != -1:
            #    if hint['rank'] == 0:# or hint['rank'] == 1:
            #        if any((f == hint['rank'] for f in fireworks.values())):
            #            return {'action_type': 'PLAY', 'card_index': card_index}

        # Check if it's possible to hint a card to your colleagues.
        if observation['information_tokens'] > 0:
            # Check if there are any playable cards in the hands of the colleagues.
            for player_offset in range(1, observation['num_players']):# This loop excludes player 0 (i.e. excludes ourself)
                player_hand: KnownHand = observation['observed_hands'][player_offset] # This is our colleague's actual hand contents.
                player_hints: OwnHand = observation['card_knowledge'][player_offset] # This is our colleague's state of knowledge about their own hand
                # Check if the card in the hand of the colleague is playable.

                #print(player_hand)
                #print(player_hints)

                for card, hint in zip(player_hand, player_hints):
                    if self.playable_card(card, fireworks) and hint['rank'] is None:
                        # Give our colleague a hint
                        return {
                            'action_type': 'REVEAL_RANK',
                            'rank': card['rank'],
                            'target_offset': player_offset
                        }
                for card, hint in zip(player_hand, player_hints):
                    if self.playable_card(card, fireworks) and hint['color'] is None:
                        # Give our colleague a hint
                        return {
                            'action_type': 'REVEAL_COLOR',
                            'color': card['color'],
                            'target_offset': player_offset
                        }

        # If no card is hintable, then discard or play.
        if observation['information_tokens'] < self.max_information_tokens:
            return {'action_type': 'DISCARD', 'card_index': 0}
        else:
            return {'action_type': 'PLAY', 'card_index': 0}
