import numpy as np

import skopt
import pickle

import ac
import gym
from State_LFA import state_lfa

env = gym.envs.make('MountainCarContinuous-v0')
phi = state_lfa(env)

def main(params):
    policy_lr, value_lr, lamb, exp_buff_len, gamma = params
    
    agent = ac.ActorCritic(env, 
    			phi, exp_buffer_length=100,episodes=1000, gamma=gamma, display=False, lamb=lamb, 
    			policy_lr=policy_lr, value_lr=value_lr)
    loss = agent.run()
    print("Loss = %f @ params = %s"%(loss, str(params)))
    return -loss

if __name__ == "__main__":
    params = [
        np.logspace(-4, -1, 10),
        np.logspace(-4, -1, 10),
        np.logspace(-5, -1, 10),
        np.linspace(10, 100, 10, dtype=int)
        (0.90, 0.99)
    ]

    res = skopt.gp_minimize(func=main, dimensions=params, n_calls=100, verbose=True)

    print(res.x, res.fun)
    pickle.dump(res, open('res.pkl', 'wb'))
