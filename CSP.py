from mne.cov import _regularized_covariance
import numpy as np
from scipy import linalg

class CSP():
	def __init__(self, n_components):
		self.n_components = n_components
		
	def compute_cov_matrix(self, X, y):
		_, n_channels, _ = X.shape
		covs = []
		for this_class in self._classes:
			x_class = X[y == this_class]
			x_class = np.transpose(x_class, [1, 0, 2])
			x_class = x_class.reshape(n_channels, -1)
			cov = _regularized_covariance(x_class)
			covs.append(cov)

		return np.stack(covs)

	def fit(self, X, y):
		# get classes
		self._classes = np.unique(y)
		# compute covs matrices
		covs = self.compute_cov_matrix(X, y)
		# decompose covs to eigen vectors and values
		eigval, eigvec = linalg.eigh(covs[0], covs.sum(0))
		# sort components
		ix = np.argsort(np.abs(eigval - 0.5))[::-1]
		eigvec = eigvec[:, ix]
		# get filters
		self.filters_ = eigvec.T
		# pick filters to keep
		pick_filters = self.filters_[:self.n_components]
		X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])
		# compute features (mean power)
		X = (X ** 2).mean(axis=2)
		# standardize features
		self.mean_ = X.mean(axis=0)
		self.std_ = X.std(axis=0)
		
		return self
    
	def transform(self, X):
		
		pick_filters = self.filters_[:self.n_components]
		X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

		X = (X ** 2).mean(axis=2)
		
		return X

	def fit_transform(self, X, y):

		self.fit(X, y)

		return self.transform(X)