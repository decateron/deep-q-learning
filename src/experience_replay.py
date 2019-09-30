"""
Module: 
    experience_replay.py
Overview:
    This module creates the Experience Replay. It helps to handle two big problems.
    The first is forgetting previous experiences.
    The second is the correlation between experiences.
Classes:
    Memory
"""



import random
import numpy as np
from collections import deque  # A deque (double ended queue) is a data type
                               # that removes the oldest element each time that you add a new element.



class Memory():
    """
    Replay buffer stores experience tuples while interacting with the environment,
    and then we sample a small batch of tuples to feed our neural network.
    """

    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Return a random batch of tuples from the buffer.

        :param batch_size: The size of the batch that will be returned.
        :return: the random batch of tuples
        """

        batch = np.random.choice(self.buffer, size=batch_size, replace=False)
        
        return batch
    
    def instantiate_memory(self, env, possible_actions, pretrain_length=64, use_commands=False):
        """
        Here we'll deal with the empty memory problem:
        we pre-populate our memory by taking random actions and storing the experience(state, action, reward, next_state)
        
        :param env: Our current game environment.
        :param possible_actions: The actions that we can take in the current environment.
        :param pretrain_length: How many experiences we want to pretrain ?
        :param use_commands: Do we want to use commands ?
        """
        for i in range(pretrain_length):
            # If it's the first step
            if i == 0:
                state = env.reset()

                state, stacked_frames = stack_frames(state, True)
            
            # Get the next_state, the rewards, done by taking a random action
            action = random.choice(possible_actions)

            if use_commands:
                # Change from an action to command
                command = change_action(action)
                next_state, reward, done, _ = env.step(command)
            else:
                next_state, reward, done, _ = env.step(action)
            
            # Stack the frames
            next_state, stacked_frames = stack_frames(next_state, False)

            
            # If the episode is finished
            if done:
                # We finished the episode
                next_state = np.zeros(state.shape)

                # Add experience to memory
                self.add((state, action, reward, next_state, done))

                # Start a new episode
                state = env.reset()

                # Stack the frames
                state, stacked_frames = stack_frames(state, True)

            else:
                # Add experience to memory
                self.add((state, action, reward, next_state, done))

                # Our new state in now the next_state
                state = next_state
    
