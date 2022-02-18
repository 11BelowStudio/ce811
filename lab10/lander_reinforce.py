# need "pip3 install numpy tensorflow gym Box2D" to run
# This code initially came from https://github.com/gabrielgarza/openai-gym-policy-gradient
# Then modified to work with TensorlowV2.x by M. Fairbank, with many further enhancements.

import gym
import matplotlib.pyplot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Tuple
import math

def calculate_reinforce_gradient1(trajectory, total_reward, baseline, action_choices):
    # This function is meant to calculate (dL/d Theta), where L=(\sum_t (log(P_t))(R-b).
    # You need to use the functions tf.math.log and tf.reduce_sum in here, plus the multiplication and subtraction operations.  
    with tf.GradientTape() as tape:
        tape.watch(table_action_probabilities_before_softmax)
        trajectory_action_probabilities=get_action_probabilities_for_trajectory(trajectory) # this returns a tensor of shape [trajectory_length,4]
        chosen_probabilities=tf.gather(trajectory_action_probabilities, indices=action_choices, axis=1, batch_dims=1) # this returns a tensor of shape [trajectory_length]
        log_probabilities=tf.math.log(chosen_probabilities)
        L=tf.reduce_sum(log_probabilities)*(total_reward-baseline)
    assert len(L.shape)==0 # checking the original large array has gone through a reduce_sum
    grads = tape.gradient(L, table_action_probabilities_before_softmax) # This calculates the gradient required by REINFORCE
    # This function doesn't actually do the update.  It just calculates the gradient ascent direction, and returns it!
    return grads
    
def calculate_reinforce_gradient(episode_observations, rewards_minus_baseline, episode_actions, policy_network):
    # This function is meant to calculate (dL/d Theta), where L=(\sum_t (log(P_t))(R-b).

    # Train on episode
    batch_trajectory=np.stack(episode_observations)
    batch_action_choices=np.stack(episode_actions).astype(np.int32)
    
    # Check all input arrays are the correct shape...
    assert batch_trajectory.shape[0]==rewards_minus_baseline.shape[0]
    assert batch_trajectory.shape[0]==batch_action_choices.shape[0]
    assert len(batch_trajectory.shape)==2
    assert len(rewards_minus_baseline.shape)==1
    assert len(batch_action_choices.shape)==1
    
    # This is the REINFORCE gradient calculation
    with tf.GradientTape() as tape:
        # Note, don't need a tape.watch here because tensorflow by default always "watches" all Variable tensors, i.e. all of our neural network weights.
        trajectory_action_probabilities=policy_network(batch_trajectory)
        # Note that the next 2 lines could be repaced by a single call to tf.keras.losses.SparseCategoricalCrossentropy
        chosen_probabilities=tf.gather(trajectory_action_probabilities,indices=batch_action_choices,axis=1, batch_dims=1) # this returns a tensor of shape [trajectory_length]
        log_probabilities=tf.math.log(chosen_probabilities)
        logprobrewards=log_probabilities*rewards_minus_baseline # Instead of using R-baseline, we are using R_t-baseline here, 
                                                                # i.e. where R_t is the reward to go from step t
        L=tf.reduce_sum(logprobrewards) 
    assert len(L.shape)==0 # checking the original large array has gone through a reduce_sum
    grads = tape.gradient(L, policy_network.trainable_weights) # This calculates the gradient required by REINFORCE
    # This function doesn't actually do the update.  It just calculates the gradient ascent direction, and returns it!
    return grads


def calculate_accumulated_discounted_rewards(episode_rewards, discount_factor) -> np.ndarray:
    #discounted_episode_rewards= np.ones_like(episode_rewards)*sum(episode_rewards)

    discounts = list(discount_factor ** i for i in range(len(episode_rewards)))

    return np.array(
        list(
            sum(episode_rewards[i + j] * (discounts[j]) for j in range(0, len(episode_rewards) - i))
            for i in range(len(episode_rewards))
        )
    )
    # We need to return a numpy array of the same length and shape as the input array episode_rewards.
    

def run_stochastic_policy(policy_network, observation):
    # Reshape observation to (1,num_features)
    observation = observation[np.newaxis,:]
    # Run forward propagation to get softmax probabilities
    action_probabilities = policy_network(observation).numpy().reshape(-1)
    # Select action using a biased sample
    # this will return the index of the action we've sampled
    action = np.random.choice(range(len(action_probabilities)), p=action_probabilities)
    return action


environment_name='LunarLander-v2'

env = gym.make(environment_name)
env = env.unwrapped

# Policy gradient has high variance, seed for reproducability
env.seed(1)

print("env.action_space", env.action_space)
print("env.observation_space", env.observation_space)
print("env.observation_space.high", env.observation_space.high)
print("env.observation_space.low", env.observation_space.low)


NUM_EPISODES = 1000
rewards = []

learning_rate=0.02/4
discount_factor=0.98
optimizer=keras.optimizers.Adam(learning_rate)        
num_inputs=env.observation_space.shape[0]
num_outputs=env.action_space.n

# Build a keras neural-network for the stochastic policy network:
hidden_nodes=[6,6]
inputs = keras.Input(shape=(num_inputs,))
x=inputs
x_ = layers.Dense(hidden_nodes[0], activation="tanh")(x)
x=tf.concat([x,x_],axis=1) # This passes shortcut connections from all earlier layers to this next one
x_ = layers.Dense(hidden_nodes[1], activation="tanh")(x)
x=tf.concat([x,x_],axis=1) # This passes shortcut connections from all earlier layers to this next one
outputs = layers.Dense(num_outputs, activation="softmax")(x)
policy_network = keras.Model(inputs=inputs, outputs=outputs, name="policy_network")
# Using the shortcut connections above means I don't need to worry 
# too much about how many hidden layers to add.  For example, if hidden 
# layers 1 and 2 are not needed then they can simply be skipped over.
policy_network.compile(optimizer='adam', loss=None)
# force the network to compile with dummy loss function.  This is so that save_model will work without warnings.

print(policy_network.summary())

reward_history=[]
mean_discounted_reward_history=[]
plt.ion()
fa: Tuple[plt.Figure, plt.Axes] = plt.subplots(nrows=1, ncols=1, figsize=(16,9))
fig: plt.Figure = fa[0]
#fig.suptitle("Reward vs Iteration")
ax: plt.Axes = fa[1]
fig.show()
plt.pause(0.001)
#plt.draw()
#plt.pause(0.001)





for episode in range(NUM_EPISODES):
    observation = env.reset()
    episode_reward = 0
    done = False
    steps=0
    episode_observations=[]
    episode_actions=[]
    episode_rewards=[]
    while not done:
        if episode%50==0: env.render()

        # 1. Choose an action based on observation
        action = run_stochastic_policy(policy_network, observation)

        # 2. Take action in the environment
        observation_, reward, done, info = env.step(action)
        steps+=1
        
        while steps>400 and not done:
            # Since REINFORCE requires trajectories to terminate, we need to 
            # need to Finish this time-wasting episode.
            # So switch engine off after 400 time steps - this will 
            # make lander crash eventually, and trajectory will terminate
            action_=0 # Switches engine off
            _, r, done, _ = env.step(action_)
            reward+=r # Put all these final sub-rewards into the final (massive) time-step's reward.
            steps+=1

        # 4. Store transition for training
        episode_observations.append(observation)
        episode_rewards.append(reward)
        episode_actions.append(action)
        # Save new observation
        observation = observation_
        
    episode_rewards_sum = sum(episode_rewards)
    rewards.append(episode_rewards_sum)
    max_reward_so_far = np.amax(rewards)
    lowest_reward_so_far = np.amin(rewards)

    print("==========================================")
    print("Episode: ", episode)
    print("Steps:",steps)
    print("Reward: ", episode_rewards_sum)
    print("Max reward so far: ", max_reward_so_far)
    reward_history.append(episode_rewards_sum)
    # 5. Train neural network
    # Discount and normalize episode reward
    
    
    if len(mean_discounted_reward_history)>10:
        hist_len=min(len(mean_discounted_reward_history),50)
        arr=np.array(mean_discounted_reward_history[-hist_len:])
        baseline=np.mean(arr) # This our estimate of a good BASELINE to be used in the REINFORCE algorithm
        # The baseline we've used here is a moving average, but really should be a fixed quantity for REINFORCE algorithm derivation.
        # Hopefully by averaging over 50 NUM_EPISODES the moving baseline should be fairly stable
        reward_scaler=1/np.std(arr) #This attempts to rescale the rewards so that they are closer to being in the range [-1,1], which 
        # should make the learning rate more appropriate
    else:
        reward_scaler=0
        baseline=0
    discounted_episode_rewards = calculate_accumulated_discounted_rewards(episode_rewards, discount_factor)
    mean_discounted_reward_history.append(np.mean(discounted_episode_rewards))
    discounted_episode_rewards -= baseline
    discounted_episode_rewards *= reward_scaler
    grads=calculate_reinforce_gradient(episode_observations, discounted_episode_rewards, episode_actions, policy_network)
    grads=[-g for g in grads]  # Put a minus sign before all of the gradients - because in RL we are trying to MAXIMISE a rewards, but optimizer.apply_graidents only works with MINIMISATION.
    optimizer.apply_gradients(zip(grads, policy_network.trainable_weights)) # This updates the parameter vector

    if (episode + 1)%50==0:
        # every 50 NUM_EPISODES, re-plot the graph
        print(reward_history)
        ax.clear()
        ax.plot(reward_history, 'b-')
        ax.axis([0, NUM_EPISODES + 1, min(-400, math.floor(lowest_reward_so_far/100)*100), max(400, math.ceil(max_reward_so_far/100)*100)])
        ax.set_ylabel('Reward')
        ax.set_xlabel('Iteration')
        ax.grid(visible=True, which="major", axis="y", linestyle="-")
        ax.grid(visible=True, which="minor", axis="y", linestyle=":")
        #ax.plot(reward_history, 'b-')
        ax.set_title("Learning Rate:" + str(learning_rate) + " Discount Factor:" + str(discount_factor))
        #ax.redraw_in_frame()
        fig.show()
        if episode != NUM_EPISODES-1:
            plt.pause(0.001)
    #
    #    ax.clear()
    #    ax.plot(reward_history, 'b-')
    #    plt.show()
    #    plt.pause(0.001)
plt.show()
fig.add_axes(ax)
fig.savefig("Result_"+datetime.now().strftime("%Y%m%d_%H%M%S")+".png")
final_average_fitness=np.array(reward_history[-40:]).mean()
print("Finished Training.  Final average fitness",final_average_fitness)
# Save the current model into a local folder
keras.models.save_model(policy_network, "Model1"+environment_name+".h5",save_format='h5')

if (input("Play demo? (Y/N)")).upper()=="Y":
    for demo in range(10):
        observation = env.reset()
        episode_reward = 0
        done = False
        steps=0
        while not(done):
            env.render()
            action = run_stochastic_policy(policy_network, observation)
            observation_, reward, done, info = env.step(action)
            episode_reward+=reward
            steps+=1
            while steps>400 and not done:
                action_=0
                _, reward, done, _ = env.step(action_)
                episode_reward+=reward
                steps+=1
            observation = observation_
