import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.linalg import inv, solve


def NSSE(xi, xj, amp, scale):
    def sigma(x):
        return 1.
    def l(x):
        return 1

    return sigma(xi)*sigma(xj)*np.sqrt(2*l(xi)*l(xj)/(l(xi)**2 + l(xj)**2))*np.exp(-(xi - xj)**2/(l(xi)**2 + l(xj)**2))


def kernel_SE(amp, scale, X, Y=None, jitter=1e-5):
    if Y is None:
        dists = squareform(pdist(X[:, None], metric="sqeuclidean"))
        K = amp*amp*np.exp(-0.5*dists/scale**2)
        # np.fill_diagonal(K, 1)
        K += jitter*np.eye(K.shape[0], K.shape[1])
    else:
        dists = cdist(X[:, None], Y[:, None], metric="sqeuclidean")
        K = amp*amp*np.exp(-0.5*dists/scale**2)
        K += jitter*np.eye(K.shape[0], K.shape[1])
    return K


def kernel_RQ(amp, scale, alpha, X, Y=None, jitter=1e-5):
    if Y is None:
        dists = squareform(pdist(X[:, None], metric="sqeuclidean"))
        tmp = dists / (2 * alpha * scale**2)
        base = 1 + tmp
        K = amp*amp*base**(-alpha)
        # np.fill_diagonal(K, 1)
        K += jitter*np.eye(K.shape[0], K.shape[1])
    else:
        dists = cdist(X[:, None], Y[:, None], metric="sqeuclidean")
        K = amp*amp*(1 + dists / (2 * alpha * scale**2)) ** (-alpha)
    return K


def make_GP_RQ_predictions(x_pred, x, y, amp, scale, alpha, sigma=None):
    """
    :note:
        y_pred = np.random.multivariate_normal(mu_pred, cov_pred)
    """
    if sigma is None:
        K = kernel_RQ(amp, scale, alpha, x)
    else:
        K = kernel_RQ(amp, scale, alpha, x, jitter=sigma**2)
    inv_K = inv(K)
    K_ = kernel_RQ(amp, scale, alpha, x, x_pred)
    K__ = kernel_RQ(amp, scale, alpha, x_pred, x_pred)

    mu = K_.T @ inv_K @ y
    cov = K__ - K_.T @ inv_K @ K_

    return mu, cov


def make_GP_SE_predictions(x_pred, x, y, amp, scale, sigma=None):
    """
    :note:
        y_pred = np.random.multivariate_normal(mu_pred, cov_pred)
    """
    if sigma is None:
        K = kernel_SE(amp, scale, x)
    else:
        K = kernel_SE(amp, scale, x, jitter=sigma**2)
    inv_K = inv(K)
    K_ = kernel_SE(amp, scale, x, x_pred)
    K__ = kernel_SE(amp, scale, x_pred, x_pred)

    mu = K_.T @ inv_K @ y
    cov = K__ - K_.T @ inv_K @ K_

    return mu, cov
