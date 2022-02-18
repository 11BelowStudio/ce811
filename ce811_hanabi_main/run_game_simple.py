import os

from hanabi_learning_environment import rl_env
from simple_agent import SimpleAgent as OurAgent
from typing import Dict, List, Tuple, Union

num_players=3 # can change this as required to 2,3,4 or 5.
environment=rl_env.make(
    environment_name='Hanabi-Full',
    num_players=num_players,
    pyhanabi_path=os.getcwd()+"\hanabi-learning-environment-master"
)

print(environment)

results: List[Dict[str, Union[Dict[str, int], int]]] = []

total_score: int = 0

for i in range(5):

    observations = environment.reset()
    # Build the team of players - each programmed with the same agent logic ("Mirror Mode")
    # Even though they are the same program logic for each player, they cannot exchange information
    # between each other, for example to see their own hand.
    agents = [OurAgent({'players': num_players}) for _ in range(num_players)]
    done = False
    episode_reward = 0
    while not done:
        for agent_id, agent in enumerate(agents):
            observation = observations['player_observations'][agent_id]
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