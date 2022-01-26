import os

from hanabi_learning_environment import rl_env
from simple_agent import SimpleAgent as OurAgent
from typing import Dict, List, Tuple, Union

import json

num_players=4 # can change this as required to 2,3,4 or 5.
environment=rl_env.make(
    environment_name='Hanabi-Full',
    num_players=num_players#,
    #pyhanabi_path=os.getcwd()+"\hanabi-learning-environment-master"
)

print(environment)

results: List[Dict[str, Union[Dict[str, int], int]]] = []

total_score: int = 0

def getstructure(data, tab = 0):
    if type(data) is dict:
        print(' '*tab + '{')
        for key in data:
            print(' '*tab + '  ' + key + ':')
            if not getstructure(data[key], tab+4):
                print(' '*(tab+4) + repr(data[key]))
        print(' '*tab + '}')
        return True
    elif type(data) is list and len(data) > 0:
        print(' '*tab + '[')
        getstructure(data[0], tab+4)
        print(' '*tab + '  ...')
        print(' '*tab + ']' + " len " + str(len(data)))
        return True
    else:
        return False


def _pretty_write_dict(dictionary):
    def _nested(obj, level=1):
        indentation_values = "\t" * level
        indentation_braces = "\t" * (level - 1)
        if isinstance(obj, dict):
            return "{\n%(body)s%(indent_braces)s}" % {
                "body": "".join("%(indent_values)s\'%(key)s\': %(value)s,\n" % {
                    "key": str(key),
                    "value": _nested(value, level + 1),
                    "indent_values": indentation_values
                } for key, value in obj.items() if key != "vectorized"),
                "indent_braces": indentation_braces
            }
        if isinstance(obj, list):
            return "[\n%(body)s\n%(indent_braces)s]" % {
                "body": "".join("%(indent_values)s%(value)s,\n" % {
                    "value": _nested(value, level + 1),
                    "indent_values": indentation_values
                } for value in obj),
                "indent_braces": indentation_braces
            }
        else:
            return "\'%(value)s\'" % {"value": str(obj)}

    dict_text = _nested(dictionary)
    return dict_text



for i in range(200):

    observations = environment.reset()




    print(observations['player_observations'][0]["vectorized"])

    print(getstructure(observations))
    turns: int = 0

    first = True


    # Build the team of players - each programmed with the same agent logic ("Mirror Mode")
    # Even though they are the same program logic for each player, they cannot exchange information
    # between each other, for example to see their own hand.
    agents = [OurAgent({'players': num_players}) for _ in range(num_players)]
    agents[0]._protag = True
    done = False
    episode_reward = 0
    while not done:
        turns += 1
        for agent_id, agent in enumerate(agents):
            observation = observations['player_observations'][agent_id]
            #if first:
            #    print(_pretty_write_dict(observation))
            #    first = False
            action = agent.act(observation)
            if observation['current_player'] == agent_id:
                assert action is not None
                current_player_action = action
                print("Player",agent_id,"to play")
                print("Player",agent_id,"View of cards",observation["observed_hands"])
                print("Fireworks",observation["fireworks"])
                print("Player",agent_id,"chose action",action)
                print()
            else:
                assert action is None

        # Make an environment step.
        observations, reward, done, unused_info = environment.step(current_player_action)
        if reward<0:
            reward=0 # we're changing the rules so that losing all lives does not result in the score being zeroed.
        episode_reward += reward

    print("Game over.  Fireworks",observation["fireworks"],"Score=",episode_reward)

    if turns > max_game_length:
        max_game_length = turns

    #print(_pretty_write_dict(observation))

    results.append(
        {
            "fireworks": observation["fireworks"],
            "score": episode_reward
        }
    )
    total_score += episode_reward

print("")

for r in range(len(results)):
    print("Game {}: Fireworks: {}, Score: {}".format(r, results[r]["fireworks"], results[r]["score"]))

print("Average score: {}".format(total_score/len(results)))

print(f"Max game length: {max_game_length}")