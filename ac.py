from policy import PolicyEstimator
from value import ValueEstimator
from collections import deque
import tensorflow as tf
import itertools
import numpy as np


class ActorCritic(object):

    def __init__(self, env, transformer,
                 exp_buffer_length=30,
                 episodes=100,
                 gamma=0.95,
                 display=False,
                 lamb=1e-5,
                 policy_lr=0.001,
                 value_lr=0.1):
    	tf.reset_default_graph()
        self.env = env
        self.exp_length = exp_buffer_length
        self.params = {'episodes': episodes,
                       'gamma': gamma,
                       'display': display,
                       'lamb': lamb,
                       'policy_lr': policy_lr,
                       'value_lr': value_lr}
        self.experiences = deque([], exp_buffer_length)
        self.policy = PolicyEstimator(
            env, transformer, lamb=lamb, learning_rate=policy_lr)
        self.value = ValueEstimator(env, transformer, learning_rate=value_lr)

    def _update_target(self, exp, gamma):
        target = exp[self.exp_length-1][2]
        for i in range(self.exp_length-2, -1, -1):
            target = exp[i][2] + gamma*target

        sars = exp.popleft()
        return target, sars

    def run(self):
        params = self.params
        nepisodes = params['episodes']
        gamma = params['gamma']
        blen = self.exp_length

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            stats = []

            for i_episode in range(nepisodes):
                state = self.env.reset()
                reward_total = 0
                value_target = np.zeros(blen)
                td_errors = np.zeros(blen)
                actions_taken = np.zeros(blen)
                states = np.zeros((blen, 2))

                for t in itertools.count():
                    action = self.policy.predict(state, sess)
                    new_state, r, done, _ = self.env.step(action)
                    reward_total += r

                    self.experiences.append((state, action, r, new_state))

                    if self.params['display']:
                        self.env.render()

                    if i_episode < 200:
                        if t >= blen-1:
                            target, sars = self._update_target(
                                self.experiences, gamma)

                            id_ = t % blen
                            states[id_] = sars[0]
                            actions_taken[id_] = sars[1]
                            value_target[id_] = target
                            td_errors[id_] = gamma**id_*(target - \
                                self.value.predict(sars[0], sess))

                            if t > 100:
                                self.policy.update(
                                    states, actions_taken, td_errors, sess)

                                self.value.update(states, value_target, sess)

                    if done:
                        break
                    state = new_state

                stats.append(reward_total)
                print("Episode: %d, reward: %f." % (i_episode, reward_total))

                if np.mean(stats[-100:]) > 90 and len(stats) >= 101:
                    print(np.mean(stats[-100:]))
                    print("Solved")

        return np.mean(stats[-100:])
