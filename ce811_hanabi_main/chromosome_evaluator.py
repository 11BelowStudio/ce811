# A way to evaluate RuleAgentChromosome
# The objective of this class is that it could easily be extended 
# into a genentic algorithm engine to improve chromosomes.
# M. Fairbank. October 2021.
import sys
from hanabi_learning_environment import rl_env
from rule_agent_chromosome import RuleAgentChromosome
import os, contextlib



def run(num_episodes, num_players, chromosome, verbose=False):
    """Run episodes."""
    environment = rl_env.make(
        environment_name='Hanabi-Full',
        num_players=num_players,
        pyhanabi_path=os.getcwd() + "\hanabi-learning-environment-master"
    )
    game_scores = []
    for episode in range(num_episodes):
        observations = environment.reset()
        agents = [RuleAgentChromosome({'players': num_players},chromosome) for _ in range(num_players)]
        done = False
        episode_reward = 0
        while not done:
            for agent_id, agent in enumerate(agents):
                observation = observations['player_observations'][agent_id]
                action = agent.act(observation)
                if observation['current_player'] == agent_id:
                    assert action is not None   
                    current_player_action = action
                    if verbose:
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
            
        if verbose:
            print("Game over.  Fireworks",observation["fireworks"],"Score=",episode_reward)
        game_scores.append(episode_reward)
    return sum(game_scores)/len(game_scores)




if __name__=="__main__":
    # TODO you could potentially code a genetic algorithm in here...
    num_players=4
    chromosome=[0,2,5,6]
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            result=run(25,num_players,chromosome)
    print("chromosome",chromosome,"fitness",result)


