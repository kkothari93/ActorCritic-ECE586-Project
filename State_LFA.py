from sklearn.kernel_approximation import RBFSampler
import sklearn
import sklearn.pipeline
import sklearn.preprocessing
import numpy as np

class state_lfa(object):
	"""Featurizes the state using LFA
	"""
	def __init__(self, env):

		samples = np.array([env.observation_space.sample() for _ in range(10000)])
		# normalize to zero mean and unit variance
		self.scaler = sklearn.preprocessing.StandardScaler()
		self.scaler.fit(samples)
		self.lfa = self._fit(samples)

	def _fit(self, samples):
		"""Implements a lfa suggested by Ashioto on OpenAI-gym"""
		lfa = sklearn.pipeline.FeatureUnion([
		    ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
		    ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
		    ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
		    ("rbf4", RBFSampler(gamma=0.5, n_components=100))
		])
		lfa.fit(self.scaler.transform(samples))
		return lfa

	def transform_state(self, state):
		if state.shape[0] == 2:
			scaled = self.scaler.transform([state])
		else:
			scaled = self.scaler.transform(state)
		state_lfa = self.lfa.transform(scaled)
		return state_lfa.ravel()



