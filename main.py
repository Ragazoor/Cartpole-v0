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
    
    env = gym.make('CartPole-v1')
    outdir = '/Results/ffnn_agent_run'
#    env = gym.wrappers.Monitor('LunarLander-v2', outdir)

    print env.observation_space
    print env.observation_space.high
    print env.observation_space.low
    print env.action_space
    
    
    # Hyperparams:  learning options, network structure, number iterations & steps,
    hyperparams = {}
    # ----------- Net Parameters:
    hyperparams['gamma'] = 0.99  #0.9
    hyperparams['n_input_nodes'] = 4
    hyperparams['n_hidden_nodes'] = 4 #10
    hyperparams['n_output_nodes'] = 2
    hyperparams['n_steps'] = 500
    hyperparams['seed'] = 13  # 13
    # ----------- worth playing with:  (current best settings in comments)
    hyperparams['init_net_wr'] = 0.05  # 0.05
    hyperparams['batch_size'] = 250  # 250
    hyperparams['epsilon'] = 1  # 1 - starting value
    hyperparams['epsilon_min'] = 0.1  # 0 - Need to explore alot so it doesn't stick in local max
    hyperparams['epsilon_decay_rate'] = 0.995  # 995  
#   ~.99 over 200 leaves it 0.1339 ~.995 over 500 its leaves it at 0.08
 
   # --- exploration/exploitation trade off is very important EVERYTHING IS UNCERTAIN
    hyperparams['target_net_hold_epsiodes'] = 1  # 5
    hyperparams['learning_rate'] = 0.01     # 0.05
    hyperparams['learning_rate_min'] = 1 #0.01 # 11 or 0.01
    hyperparams['learning_rate_decay'] = 0.5  # 0.5
    hyperparams['n_updates_per_episode'] = 1  # 1 - means pick X random minibatches, doing GradDescent on each
    hyperparams['nmr_decimals_tiles'] = 2 # the resolution of the tiles are 10^-1 
    hyperparams['max_memory_len'] = 500  # 500 - number of (s,a,r,s',done) tuples
    hyperparams['n_iter'] = 3000  # 1000
    hyperparams['n_episodes_per_print'] = 300
    hyperparams['net_hold_epsilon'] = 4 # 5 or 10
    hyperparams['net_hold_lr'] = 2000
    hyperparams['C'] = 1.4 # Higher values encourages exploration
    hyperparams['MCTS_max_depth'] = 0 # Determines how deep the tree may be
    # ------------ BEST SETTINGS GIVE: test mean: 200 +- 0

    # FFNN agent:
    agent = FFNNAgent(hyperparams)
    
    # starts to train agent
    agent.optimize_episodes(env, rend = True)
    agent.net.plot_error()    
    agent.plot_reward()
    plt.show()
    # test to see how it goes
    agent.epsilon = 0
    agent.n_iter = 100
    agent.n_episodes_per_print = 5
    agent.C = 0
    #agent.net.set_lr(0.0001)
    agent.optimize_episodes(env,rend = True)

    agent.net.print_params()

if __name__ == '__main__':
    main_FFNNAgent()


