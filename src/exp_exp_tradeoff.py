import numpy as np
import random



def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions, gamma, gamma_decay_rate):
    """
    Choose action a from state s using epsilon greedy strategy. Calculate exploration probability. And decrease gamma.
    
    :param explore_start: Exploration start - hyperparameter.
    :param explore_stop: Exploration start - hyperparameter.
    :param decay_rate: The rate of decreasing exploration probability.
    :param decay_step: Help the decay rate decrease exploration probability.
    :param state: The current state of our Agent
    :param possible_actions: Matrix with all possible actions in the current environment.
    :return: The current action, exploration probability and gamma
    """

    # First we randomize a float number [0, 1)
    exp_exp_tradeoff = np.random.rand()

    # Epsilon greedy strategy
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if explore_probability > exp_exp_tradeoff:
        # Make a random action (exploration)
        action = random.choice(possible_actions)
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs value state
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})

        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[choice]
    
    gamma = gamma * np.exp(-gamma_decay_rate * decay_step)

    return action, explore_probability, gamma



def action_to_command(action, action_size, commands):
    """
    Change from an action to command, because the environment takes only commands.

    :param action: The action that produces our DQN
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
    
    