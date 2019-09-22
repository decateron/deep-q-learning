"""
Module:
    environment.py
Overview:
    This module contains a dictionary with names of environments and their action spaces.
    Also, this module contains a function that creates a specified environment and returns its parameters.
Functions:
    create_environment
"""



import retro
import numpy as np



game_commands = {'SpaceInvaders-Atari2600': np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                                      [0, 0, 0, 0, 0, 0, 1, 0],
                                                      [0, 0, 0, 0, 0, 0, 0, 1]])}



def create_environment(lib_name, game, action_space, use_commands=False, game_commands=None):
    """
    Create environment. Return much useful information about the environment.

    :param lib_name: The name of the library that contains our environment.
    :param game: The name of the game we want to play.
    :param action_space: How many actions we can take in this game ?
    :param use_commands: Do we want to use commands ?
    :param game_commands: Available commands in this game.
    :return: environment object, action space, possible actions, commands.

    """

    # Create an environment
    env = lib_name.make(game)
    print("The size of our frame is: ", env.observation_space)

    # Initialize possible actions
    possible_actions = np.array(np.eye(action_space, dtype=int).tolist())

    print()
    print(possible_actions)

    # Initialize commands 
    if use_commands:
        commands = game_commands[game]
    else:
        commands = False


    return env, action_space, possible_actions, commands
