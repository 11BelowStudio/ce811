"""Simple Agent.""" 
# Modified from https://github.com/deepmind/hanabi-learning-environment/blob/master/hanabi_learning_environment/agents/simple_agent.py 

from hanabi_learning_environment.rl_env import Agent


class SimpleAgent(Agent):
  """Agent that applies a simple heuristic."""

  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    self.config = config
    # Extract max info tokens or set default to 8.
    self.max_information_tokens = config.get('information_tokens', 8)

  def playable_card(self, card, fireworks):
    """A card is playable if it can be placed on the fireworks pile."""
    return card['rank'] == fireworks[card['color']]

  def act(self, observation):
    """Act based on an observation."""
    if observation['current_player_offset'] != 0:
      return None # No action returned (because it's not our turn!)

    fireworks = observation['fireworks'] # This is a dictionary like {'R': 0, 'Y': 1, 'G': 0, 'W': 1, 'B': 0}
    
    
    card_knowledge_about_own_hand=observation['card_knowledge'][0] 
    '''
    card_knowledge_about_own_hand is a list of dictionaries, like  
        [{'color': None, 'rank': None}, {'color': None, 'rank': None}, 
        {'color': 'B', 'rank': None}, {'color': None, 'rank': None}, 
        {'color': 'W', 'rank': None}] 
    This particular card_knowledge_about_own_hand list tells us that we have 5 cards
    in our hand, we the only things we know about them is that the 3rd card is blue,
    and the 5th card is white.  We don't know any of the card ranks yet.
    '''

    # Check if there are any pending hints and play the card corresponding to
    # the hint.
    for card_index, hint in enumerate(card_knowledge_about_own_hand):
      if hint['color'] is not None or hint['rank'] is not None:
        # This card has had hint, so just play it and hope for the best...
        return {'action_type': 'PLAY', 'card_index': card_index}

    # Check if it's possible to hint a card to your colleagues.
    if observation['information_tokens'] > 0:
      # Check if there are any playable cards in the hands of the colleagues.
      for player_offset in range(1, observation['num_players']):# This loop excludes player 0 (i.e. excludes ourself)
        player_hand = observation['observed_hands'][player_offset] # This is our colleague's actual hand contents.
        player_hints = observation['card_knowledge'][player_offset] # This is our colleague's state of knowledge about their own hand
        # Check if the card in the hand of the colleague is playable.
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
