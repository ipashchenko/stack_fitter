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


def profile_boccardi(z, z_br, k_b, k_a, R_br, h):
    return R_br*2**((k_b - k_a)/h)*(z/z_br)**k_b*(1 + (z/z_br)**h)**((k_a - k_b)/h)


def sigmoid(x, x_0, alpha):
    return 1./(1. + np.exp(-(x - x_0)/alpha))

def profile_sigmoid(z, z_0, z_1, z_br, k_b, k_a, b_b, dz):
    f1 = b_b*(z + z_0)**k_b
    b_a = b_b*(z_br + z_0)**k_b/(z_br + z_1)**k_a
    f2 = b_a*(z + z_1)**k_a
    sigma = sigmoid(z, z_br, dz)
    return (1 - sigma)*f1 + sigma*f2


def profile_teared_sigmoid(z, z_0, z_1, z_br, k_b, k_a, b_b, dz):
    b_a = b_b*(z_br + z_0)**k_b/(z_br + z_1)**k_a
    sigma = sigmoid(z, z_br, dz)
    f1 = (1-sigma)*b_b*(z + z_0)**k_b
    f2 = sigma*b_a*(z + z_1)**k_a
    before = z < z_br
    result = f2
    result[before] = f1[before]
    return result


def profile_1(z, z_0, z_1, z_br, k_b, k_a, b_b, dz):

    def func(z, b, shift, k):
        return b*(z + shift)**k

    def func_der(z, b, shift, k):
        return k*b*(z + shift)**(k - 1)

    f1 = func(z, b_b, z_0, k_b)
    b_a = b_b*(z_br + z_0)**k_b/(z_br + z_1)**k_a
    f2 = func(z, b_a, z_1, k_a)

    z_before = z_br - 0.5*dz
    z_after = z_br + 0.5*dz
    C1 = func(z_before, b_b, z_0, k_b)
    C2 = func(z_after, b_a, z_1, k_a)
    C3 = func_der(z_before, b_b, z_0, k_b)
    C4 = func_der(z_after, b_a, z_1, k_a)
    b = np.array([C1, C2, C3, C4])

    A = np.array([[z_before**3, z_before**2, z_before, 1],
                  [z_after**3, z_after**2, z_after, 1],
                  [3*z_before**2, 2*z_before, 1, 0],
                  [3*z_after**2, 2*z_after, 1, 0]])

    coeffs = np.linalg.solve(A, b)
    print(coeffs)
    poly = np.polynomial.polynomial.polyval(z, coeffs[::-1])

    before = z < z_before
    after = z > z_after
    between = np.logical_and(z > z_before, z <= z_after)
    result = f2
    result[before] = f1[before]
    result[between] = poly[between]

    # fig, axes = plt.subplots(1, 1)
    # axes.axvline(z_br, color="k")
    # axes.plot(z[before], result[before])
    # axes.plot(z[between], result[between])
    # axes.plot(z[after], result[after])
    # plt.show()

    return result


def profile_rigorous(z, z_0, z_1, z_br, k_b, k_a, b_b):
    f1 = b_b*(z + z_0)**k_b
    b_a = b_b*(z_br + z_0)**k_b/(z_br + z_1)**k_a
    f2 = b_a*(z + z_1)**k_a
    before = z < z_br
    result = f2
    result[before] = f1[before]
    return result


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")
    z = np.linspace(0.5, 30, 1000)
    k_b = 1.0
    k_a = 0.25
    z_br = 7.0
    R_br = 3.0
    z_0 = -0.15
    z_1 = 1.
    b_b = 0.5
    h = 100
    dz = 0.5
    fig, axes = plt.subplots(1, 1, figsize=(6.4*2, 4.8*2))
    # plt.plot(z, profile_boccardi(z, z_br, k_b, k_a, R_br, h))
    R_true = profile_rigorous(z, z_0, z_1, z_br, k_b, k_a, b_b)
    R_sigmoid = profile_sigmoid(z, z_0, z_1, z_br, k_b, k_a, b_b, dz)
    R_poly = profile_1(z, z_0, z_1, z_br, k_b, k_a, b_b, dz)
    # R_teared_sigmoid = profile_teared_sigmoid(z, z_0, z_1, z_br, k_b, k_a, b_b, dz)
    axes.plot(z, R_true, lw=2, color="C0", label="True")
    axes.plot(z, R_sigmoid, lw=2, color="C1", label="Sigmoid")
    axes.plot(z, R_poly, lw=2, color="C2", label="poly")
    # axes.plot(z, R_teared_sigmoid, lw=2, color="C2", label="T.Sigmoid")
    axes.plot(z, R_sigmoid-R_true, label=r"$\Delta$ Sigmoid", ls="--", color="C1")
    axes.plot(z, R_poly-R_true, label=r"$\Delta$ poly", ls="--", color="C2")
    # axes.plot(z, R_teared_sigmoid-R_true, label=r"$\Delta$ T.Sigmoid", ls="--", color="C2")
    axes.axhline(0.0, color="k")
    axes.set_xlim([z_br - 4., z_br + 4.])
    plt.legend()
    plt.show()

