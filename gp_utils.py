import warnings
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.linalg import inv, solve
import matplotlib.pyplot as plt


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


def kernel_linear(sigma_b, sigma_v, c, X, Y=None, jitter=1e-5):
    if Y is None:
        dists = sigma_v*sigma_v*squareform(pdist((X - c)[:, None], metric="sqeuclidean"))
        K = dists + sigma_b*sigma_b
        # np.fill_diagonal(K, 1)
        K += jitter*np.eye(K.shape[0], K.shape[1])
    else:
        dists = sigma_v*sigma_v*cdist((X-c)[:, None], (Y-c)[:, None], metric="sqeuclidean")
        K = dists + sigma_b*sigma_b
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

def make_GP_SElin_predictions(x_pred, x, y, amp, scale, amp_b, amp_v, c, sigma=None):
    """
    :note:
        y_pred = np.random.multivariate_normal(mu_pred, cov_pred)
    """
    if sigma is None:
        K = kernel_SE(amp, scale, x)
        K_lin = kernel_linear(amp_b, amp_v, c, x)
    else:
        K = kernel_SE(amp, scale, x, jitter=sigma**2)
        K_lin = kernel_linear(amp_b, amp_v, c, x)
    K = K@K_lin
    # #r x #r
    inv_K = inv(K)

    # #r_pred x #r
    K_ = kernel_SE(amp, scale, x_pred, x)
    K_lin_ = kernel_linear(amp_b, amp_v, c, x_pred, x)
    # #r_pred x #r
    K_ = K_ * K_lin_

    # #r_pred x #r_pred
    K__ = kernel_SE(amp, scale, x_pred, x_pred)
    # #r_pred x #r_pred
    K_lin__ = kernel_linear(amp_b, amp_v, c, x_pred, x_pred)
    # #r_pred x #r_pred
    K__ = K__ * K_lin__

    mu = K_ @ inv_K @ y
    cov = K__ - K_ @ inv_K @ K_.T

    return mu, cov


def gp_dist_plot(samples, x, ax, palette, fill_alpha=0.1, samples_alpha=0.1, plot_samples=True, fill_kwargs=None,
                 samples_kwargs=None):
    """

    :param samples:
        Numpy array with shape (#samples, #x), where #x - number of data points for predictions and #samples - number
        of predicted samples.
    :param x:
        x-coordinates for predictions.
    :param ax:
        Matplotlib Axes instance.
    :param palette:
        "Reds", "Blues", ...
    fill_kwargs: dict
        Additional arguments for posterior interval fill (fill_between).
    samples_kwargs: dict
        Additional keyword arguments for samples plot.
    """
    if fill_kwargs is None:
        fill_kwargs = {}
    if samples_kwargs is None:
        samples_kwargs = {}
    if np.any(np.isnan(samples)):
        warnings.warn(
            "There are `nan` entries in the [samples] arguments. "
            "The plot will not contain a band!",
            UserWarning,
        )
    cmap = plt.get_cmap(palette)
    percs = np.linspace(51, 99, 40)
    colors = (percs - np.min(percs)) / (np.max(percs) - np.min(percs))
    samples = samples.T
    x = x.flatten()
    for i, p in enumerate(percs[::-1]):
        upper = np.percentile(samples, p, axis=1)
        lower = np.percentile(samples, 100 - p, axis=1)
        color_val = colors[i]
        ax.fill_between(x, upper, lower, color=cmap(color_val), alpha=fill_alpha, **fill_kwargs)
    if plot_samples:
        # plot a few samples
        idx = np.random.randint(0, samples.shape[1], 30)
        ax.plot(x, samples[:, idx], color=cmap(0.9), lw=1, alpha=samples_alpha, **samples_kwargs)

    return ax
