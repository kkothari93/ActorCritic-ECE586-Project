import os
import gym
from ac import ActorCritic
from State_LFA import state_lfa


# writing video to a directory
env = gym.envs.make("MountainCarContinuous-v0")
video_dir = os.path.abspath("./videos")
if not os.path.exists(video_dir):
    os.makedirs(video_dir)
# force = True will clear all previous files
env = gym.wrappers.Monitor(env, video_dir, force=True)


if __name__ == "__main__":
	# parameters obtained after hyperparameter optimization
    policy_lr, value_lr, lamb, gamma = [0.00046415888336127773,
                                        0.021544346900318822,
                                        0.035938136638046257,
                                        0.98999999999999999]
    phi = state_lfa(env) # state kernelizer

    # initialize agent with given parameters
    agent = ActorCritic(env, phi, exp_buffer_length=100, episodes=1000,
                        gamma=gamma, display=False, lamb=lamb,
                        policy_lr=policy_lr, value_lr=value_lr)
    loss = agent.run()
    print(-loss)
    env.close()
