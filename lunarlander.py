# COMP532 CA2 Lunar Lander Game

# import libraries
import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

from keras import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.activations import relu, linear

# set seed (optional, uncomment if desired) and create lunar lander environment
env = gym.make('LunarLander-v2')
# rseed = 42
# env.seed(rseed)
# random.seed(rseed)
# np.random.seed(rseed)

# set global parameters 
input_nodes = 150
hidden_nodes = 120
lr = 0.001
state_sps = env.observation_space.shape[0]
acts = env.action_space.n
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay = 0.008
epochs = 500
discount_factor = 0.99

"""
Functions
"""

def nn_model(state_space, actions):
    '''
    Function to compile a neural network with 3 layers (input, 1 hidden, output).

    Parameters:
        state_space: int - the obersvation space of the environment (value is 8 for lunar lander game)
        actions: int - the number of actions available to the agent (value is 4 for the lunar lander discrete game)

    Return:
        model: tensorflow keras neural network - the compiled neural network
    '''

    model = Sequential()
    model.add(Dense(input_nodes, input_dim=state_space, activation=relu))
    model.add(Dense(hidden_nodes, activation=relu))
    model.add(Dense(actions, activation=linear))
    model.compile(loss='mse', optimizer=Adam(learning_rate=lr))

    return model

def update_model(env, mem_deque, model, target_model, done):
    '''
    Function to update and refit neural network model based on Q-values.

    Parameters:
        env: openai gym environment - game environment
        mem_deque: deque - deque of state/action information
        model: tensorflow keras neural network - compiled neural network
        target_model: tensorflow keras neural network - target neural network
        done: Boolean - if the episode is done

    Return:
        none: function does not return anything, only fits neural network model based on Q-values
    '''
    
    # logic to end episode if game reaches 1000 steps
    min_replay_size = 1000
    if len(mem_deque) < min_replay_size:
        return

    # set batch size and create mini batch of that size
    batch_size = 128 
    mini_batch = random.sample(mem_deque, batch_size)

    # pull out current states, current Q-values, next states, and next Q-values
    current_states = np.array([obs[0] for obs in mini_batch])
    current_qvals_lst = model.predict(current_states)
    next_states = np.array([obs[3] for obs in mini_batch])
    next_qvals = target_model.predict(next_states)

    # create lists to store states and Q-values
    states = []
    qvals = []

    # update Q-value - action pairs for each set of values in the mini batch according to the Bellman equation 
    for idx, (observation, action, reward, next_obs, done) in enumerate(mini_batch):
        if not done:
            max_qval = reward + discount_factor * np.max(next_qvals[idx])
        else:
            max_qval = reward
        
        current_qvals = current_qvals_lst[idx]
        current_qvals[action] = (1 - lr) * current_qvals[action] + lr * max_qval

        states.append(observation)
        qvals.append(current_qvals)

    # refit neural network model with updated Q-values
    model.fit(np.array(states), np.array(qvals), batch_size=batch_size, verbose=0, shuffle=True)

"""
Train Agent
"""

# initialize first neural networks
model = nn_model(state_sps, acts)

target_model = nn_model(state_sps, acts)
target_model.set_weights(model.get_weights())

# create deque that will save state/action information
mem_deque = deque(maxlen=50000)

# create rewards list to store episode rewards
rewards = []

update_target_model = 0

# train agent over defined number of episodes/epochs
for epoch in range(epochs):
    episode_rewards = 0
    state = env.reset()
    done = False

    # while loop for each episode that ends when lander crashes/lands
    while not done:
        update_target_model += 1
        env.render()

        if np.random.rand() <= epsilon:
            # explore action
            action = env.action_space.sample()
        
        else:
            # exploit action
            reshape_state = state.reshape([1, state.shape[0]])
            pred = model.predict(reshape_state).flatten()
            action = np.argmax(pred)
        
        # perform action
        next_state, reward, done, info = env.step(action)

        # save state/action information
        mem_deque.append([state, action, reward, next_state, done])
        
        # update target neural network every 4 episodes
        if update_target_model % 4 == 0 or done:
            update_model(env, mem_deque, model, target_model, done) # check and see if same name vars
        
        # update state 
        state = next_state

        # add reward to running episode total
        episode_rewards += reward
        
        # when episode is done, print results and append reward total to list
        if done:
            print(f'episode done after {update_target_model} steps')
            print(f'total episode reward: {episode_rewards}')
            print(f'final reward: {reward}')
            print(f'epoch: {epoch}')
            rewards.append(episode_rewards)

            # if episode last more than 100 actions, update target model weights
            if update_target_model >= 100:
                target_model.set_weights(model.get_weights())
                update_target_model = 0
            break
    
    # decay epsilon exploration rate
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * epoch)
    
    # escape logic to end while loop once game is solved
    if np.mean(rewards[-100:]) > 200:
        print(f'Game solved at {epoch} iterations.')
        print(f'Average reward: {np.mean(rewards[-100:])}')
        break
    
env.close()

"""
Visualization
"""

# visualize results
plt.plot([n for n in range(0, len(rewards))], rewards)
plt.title('Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()