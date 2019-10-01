"""
Module:
    change_action.py
Overview:
    This module helps to handle the action space problem.
    Different environments have different actions and action size.
    That's why I need to create a more general interface and reduce ambiguity.
Functions:
    action_to_command
"""



import numpy as np



def action_to_command(action, action_size, commands):
    """
    Change from an action to command, because the environment takes only commands.
    :param action: The action that produces our DQN.
    :param action_size: How many actions we have in this environment ?
    :param commands: The list of commands.
    :return: The command
    """

    # Change from an array to a list
    action = list(action)  

    # Get the index of 1
    index = action.index(1)

    # Choose the comand
    if index < action_size and index == 0:
        command = commands[index]
    elif index < action_size and index == 1:
        command = commands[index]
    elif index < action_size and index == 2:
        command = commands[index]
    elif index < action_size and index == 3:
        command = commands[index]
    elif index < action_size and index == 4:
        command = commands[index]
    elif index < action_size and index == 5:
        command = commands[index]
    elif index < action_size and index == 6:
        command = commands[index]
    elif index < action_size and index == 7:
        command = commands[index]
    elif index < action_size and index == 8:
        command = commands[index]
    elif index < action_size and index == 9:
        command = commands[index]
    elif index < action_size and index == 10:
        command = commands[index]
    elif index < action_size and index == 11:
        command = commands[index]
    elif index < action_size and index == 12:
        command = commands[index]
    elif index < action_size and index == 13:
        command = commands[index]
    elif index < action_size and index == 14:
        command = commands[index]
    elif index < action_size and index == 15:
        command = commands[index]
    elif index < action_size and index == 16:
        command = commands[index]
    elif index < action_size and index == 17:
        command = commands[index]
    elif index < action_size and index == 18:
        command = commands[index]
    elif index < action_size and index == 19:
        command = commands[index]
    
    return command

