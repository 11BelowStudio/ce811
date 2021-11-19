# CE811 Hanabi

This repository introduces the Google DeepMind hanabi-learning-environment, by providing some educational starter examples for CE811 Hanabi.

### Dependencies

- pip install hanabi-learning-environment

## Getting Started

Just run the python script run_game_simple.py.

- python run_game_simple.py

Then look at the code in run_game_simple.py and  simple_agent.py and try to figure out what's going on.

## Improving run_game_simple.py

Try to upgrade run_game_simple.py so that it plays 20 games and calculates the average score.  Doing this you should see an average score of around 5 or 6, with simple_agent playing.

## Improving the SimpleAgent
Try to upgrade the simple_agent.py to play a more intelligent game.  Here are two rules to add that should improve its play:

1. The current simple_agent logic *just plays any card it has received any kind of hint for*.  Try to upgrade this logic, so that it only plays a hinted card if either:
    - The hint received specificies a playable Color AND Rank (both of which combine to make a card that is definitely "playable" onto the fireworks piles).
    - Or, the rank of the hint is "0", but no colour hint has been received.  These cards are probably playable (if the game is in its early stages).
2. The current simple_agent logic give hints to colleagues if they have a playable card that has no hint.  In this case it hints the colour.  Try to upgrade this to:
    - First, prioritise giving a hint of the *rank* of any playable card which a colleague may happen to have. 
    - Second, prioritise giving a hint of the *color* of any playable card which a colleague may happen to have. 

If both of these upgrades to simple_agent are made then the performance should improve to around 14 or 15 per game.


### Hanabi Python Engine Source Code

- The source code for Google DeepMind's is included for reference in the subfolder hanabi-learning-environment-master.  The content of this subfolder is a straight mirror of https://github.com/deepmind/hanabi-learning-environment.  That subfolder is copyright Google DeepMind.  Look in it to get more details of how the game engine works.
