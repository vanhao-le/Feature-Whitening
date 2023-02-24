import numpy as np
from scipy import linalg
from sklearn.utils import check_array, as_float_array
from sklearn.utils.validation import check_is_fitted
'''
https://github.com/devyhia/cifar-10/blob/master/zca.py

The method you should use, as always, depends on what you want :
— ZCA-whitening is the unique procedure that maximizes the average cross-covariance 
between each component of the whitened and original vectors
— PCA-whitening is the unique sphering procedure that maximizes the integration, or compression, 
of all components of the original vector X in each component of the sphered vector X' based on the cross-covariance 
as underlying measure.

If you plan on reducing the dimension of your data, use PCA-whitening. 
If you want your whitened data to be close to your original data, use ZCA whitening.
'''

class ZCA():

    def __init__(self, regularization=10**-5, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X, y=None):
        """Compute the mean, whitening and dewhitening matrices.
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data used to compute the mean, whitening and dewhitening matrices.
        """
        X = check_array(X)
        X = as_float_array(X, copy = self.copy)
        # Step 1: mean-center the data
        self.mean_ = np.mean(X, axis=0)
        X_ = X - self.mean_
        cov = np.dot(X_.T, X_) / (X_.shape[0]-1)
        U, S, _ = linalg.svd(cov)
        # Clip (limit) the values in an array.
        s = np.sqrt(S.clip(self.regularization))
        s_inv = np.diag(1./s)
        s = np.diag(s)
        self.whiten_ = np.dot(np.dot(U, s_inv), U.T)
        self.dewhiten_ = np.dot(np.dot(U, s), U.T)
        return self    

    def transform(self, X, y=None, copy=None):
        """Perform ZCA whitening
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data to whiten along the features axis.
        """
        check_is_fitted(self, 'mean_')
        X = as_float_array(X, copy=self.copy)
        return np.dot(X - self.mean_, self.whiten_.T)

    def inverse_transform(self, X, copy=None):
        """Undo the ZCA transform and rotate back to the original
        representation
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data to rotate back.
        """
        check_is_fitted(self, 'mean_')
        X = as_float_array(X, copy=self.copy)
        return np.dot(X, self.dewhiten_) + self.mean_