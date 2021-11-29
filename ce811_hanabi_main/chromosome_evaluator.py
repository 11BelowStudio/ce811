# A way to evaluate RuleAgentChromosome
# The objective of this class is that it could easily be extended 
# into a genentic algorithm engine to improve chromosomes.
# M. Fairbank. October 2021.
import random
import sys
from hanabi_learning_environment import rl_env
from rule_agent_chromosome import RuleAgentChromosome, SimpleRuleChromosome, RulesChromosome
import os, contextlib

from typing import List

num_players: int = 4
num_episodes: int = 25

environment = rl_env.make(
        environment_name='Hanabi-Full',
        num_players=num_players,
        pyhanabi_path=os.getcwd() + "\hanabi-learning-environment-master"
    )

def run(num_episodes, num_players, chromosome, verbose=False):
    """Run episodes."""
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
            print("Game over.  Fireworks", observation["fireworks"],"Score=",episode_reward)
        game_scores.append(episode_reward)
    return sum(game_scores)/len(game_scores)


def run_fitness_fun(individual: RulesChromosome) -> float:
    return run(num_episodes, num_players, individual.to_rule_priority_list)


def GA_runner(
        generations: int,
        pop_size: int,
        mut_rate: float = 0.25,
        tournament_size: int = 3,
        print_fittest_per_generation: bool = True,
        print_all_per_generation: bool = False,
        verbose: bool = True
) -> RulesChromosome:

    population: List[RulesChromosome] = []
    for _ in range(pop_size):
        population.append(RulesChromosome())
        if verbose:
            print("Initialized {} of {}".format(_+1, pop_size))
    population.sort(reverse=True)

    if print_fittest_per_generation:
        print("Initial fittest:\n{}".format(population[0]))
    elif print_all_per_generation:
        print("Initial population")
        for p in population:
            print(p)

    for g in range(generations):
        new_generation: List[RulesChromosome] = []
        while len(new_generation) < pop_size:
            if random.random() < mut_rate:
                new_generation.append(RulesChromosome.mutate(max(random.sample(population, tournament_size))))
            else:
                tourn: List[RulesChromosome] = random.sample(population, tournament_size+1)
                tourn.sort(reverse=True)
                new_generation.append(RulesChromosome.crossover(
                    tourn[0], tourn[1]
                ))

        population.extend(new_generation)

        while len(population) > pop_size:
            population.remove(min(random.sample(population, tournament_size)))
        population.sort(reverse=True)

        if print_fittest_per_generation:
            print("Generation {} fittest:\n{}\n".format(g, population[0]))
        elif print_all_per_generation:
            print("Generation {} population".format(g))
            for p in population:
                print(p)
            print("")

    return max(population)

if __name__=="__main__":
    # TODO you could potentially code a genetic algorithm in here...
    num_players=4

    RulesChromosome.define_fitness_function(run_fitness_fun)

    chromosome: List[int] = [0,2,5,6]

    # best: SimpleRuleChromosome = SimpleRuleChromosome()
    best: RulesChromosome = GA_runner(30, 15, 0.25, 4, True, False)

    #with open(os.devnull, 'w') as devnull:
    #    with contextlib.redirect_stdout(devnull):
     #       #result=run(25,num_players,chromosome)
    #        best: SimpleRuleChromosome = GA_runner(30, 8, 0.25, 3, True, False)
    #print("chromosome",chromosome,"fitness",result)
    print("Best result: {}".format(best))


