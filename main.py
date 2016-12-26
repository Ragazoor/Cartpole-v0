import numpy as np
import matplotlib.pyplot as plt
import logging
import gym
from FFNNAgent import *
###########
# STRUCTURE
###########

# MAIN - Initialises FFNNAgent, trains agent, 
#   |
#   |
# FFNNAgent - Recieves feedBack from env, makes action
#   |
#   |
# FFNN - called by Agent for deciding action and training

def main_FFNNAgent():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Init and constants:
    env = gym.make('CartPole-v0')
    outdir = '/Results/ffnn_agent_run'

    # Hyperparams:  learning options, network structure, number iterations & steps,
    hyperparams = {}
    # ----------- NOT worth playing with:
    hyperparams['gamma'] = 0.95
    hyperparams['n_input_nodes'] = 4
    hyperparams['n_hidden_nodes'] = 4
    hyperparams['n_output_nodes'] = 2
    hyperparams['n_steps'] = 200
    hyperparams['seed'] = 13  # 13
    # ----------- worth playing with:  (current best settings in comments)
    hyperparams['init_net_wr'] = 0.05  # 0.05
    hyperparams['batch_size'] = 50  # 50
    hyperparams['epsilon'] = 1.0  # 1 - starting value
    hyperparams['epsilon_min'] = 0.11  # 0.1  - always keep exploring?
    hyperparams['epsilon_decay_rate'] = 0.95  # 95    # 0.98         ~.995 over 500 its leaves it at 0.08
    # -- observation is that exploration/exploitation trade off is very important
    hyperparams['target_net_hold_epsiodes'] = 1  # 1
    hyperparams['learning_rate'] = 0.15        # 0.15
    hyperparams['learning_rate_min'] = 11 # 0.001
    hyperparams['learning_rate_decay'] = 0.9  # 0.9
    hyperparams['n_updates_per_episode'] = 1  # 1 - means pick X random minibatches, doing GradDescent on each
    hyperparams['max_memory_len'] = 100  # 100 - number of (s,a,r,s',done) tuples -- ~big seems bad
    hyperparams['n_iter'] = 1000  # 1000
    # ------------------------------------  BEST SETTINGS GIVE: test mean: 200 +- 0

    # FFNN agent:
    agent = FFNNAgent(hyperparams)
    
    

    # starts to train agent
    agent.optimize_episodes(env)
    agent.net.plot_error()    
    
    print agent.net.lr
    # test to see how it goes
    agent.episode = 0
    agent.n_iter = 100
    agent.optimize_episodes(env,rend = True)
if __name__ == '__main__':
    main_FFNNAgent()


