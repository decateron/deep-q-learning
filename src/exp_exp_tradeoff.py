import numpy as np
import random



def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions, gamma_start, gamma_decay_rate, step):
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
    
    gamma = gamma_start * np.exp(-gamma_decay_rate * step)

    return action, explore_probability, gamma
