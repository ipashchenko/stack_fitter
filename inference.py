import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from astropy import units as u, constants as const, cosmology
from scipy.stats import median_abs_deviation
import functools
import matplotlib.pyplot as plt

cosmo = cosmology.WMAP9

# Speed of light [cm / s]
c = 29979245800.0
# Gravitational constant
G = 6.67430e-08
# Parsec [cm]
pc = 3.0856775814671913e+18
# Mass of the Sun [g]
M_sun = 1.98840987e+33

@functools.lru_cache()
def ang_to_dist(z):
    return cosmo.kpc_proper_per_arcmin(z)


@functools.lru_cache()
def mas_to_pc(l_mas, z, theta_obs):
    """
    Given some observed length ``l_mas`` [mas] it returns corresponding de-projected length in pc for a given redshift
    ``z`` and jet viewing angle ``theta_obs`` [rad].
    """
    return (l_mas*u.mas*ang_to_dist(z)).to(u.pc).value/np.sin(theta_obs)

@functools.lru_cache()
def get_R_g(M_BH):
    """
    Returns R_g [cm].
    """
    return G*M_BH/c/c

def get_R_horizon(M_BH, a):
    """
    Returns R_h [cm].
    """
    return get_R_g(M_BH)*(1.0 + np.sqrt(1.0 - a*a))

def get_R_L(M_BH, a):
    """
    Returns R_L [cm].
    """
    return 4.0*get_R_horizon(M_BH, a)/a

def get_z1_Ma(gamma_in, sigma_M, M_BH_sun, a, theta_jet=None):
    """
    The de-projected (real, not observed) position of the break.

    :param gamma_in:
        Initial Lorenz factor of the bulk motion.
    :param sigma_M:
        Initial magnetization.
    :param M_BH_sun:
        Black hole mass [Sun masses].
    :param a:
        Spin of the black hole.
    :param theta_jet: (optional)
        Conical half-angle [rad]. If ``None`` then use expression for the parabolic case. (default: ``None``)
    :return:
        The distance to the break position in pc.
    """
    R_L = get_R_L(M_BH_sun*M_sun, a)
    # Parabolic case
    if theta_jet is None:
        return gamma_in*sigma_M*R_L/pc
    # Conical case
    else:
        return np.sqrt(gamma_in*sigma_M)*R_L/theta_jet/pc


def get_r1_Ma(gamma_in, sigma_M, M_BH_sun, a):
    """
    The radius of the jet at the break z_1.

    :param gamma_in:
        Initial Lorenz factor of the bulk motion.
    :param sigma_M:
        Initial magnetization.
    :param M_BH_sun:
        Black hole mass [Sun masses].
    :param a:
        Spin of the black hole.
    :return:
        The radius at the break position in pc.
    """
    R_L = get_R_L(M_BH_sun*M_sun, a)
    # Parabolic case
    if theta_jet is None:
        return gamma_in*sigma_M*R_L/pc
    # Conical case
    else:
        return np.sqrt(gamma_in*sigma_M)*R_L/pc


def get_z1_RL(gamma_in, sigma_M, R_L, theta_jet=None):
    """
    The de-projected (real, not observed) position of the break.

    :param gamma_in:
        Initial Lorenz factor of the bulk motion.
    :param sigma_M:
        Initial magnetization.
    :param R_L:
        Light cylinder radius [pc].
    :param theta_jet: (optional)
        Conical half-angle [rad]. If ``None`` then use expression for the parabolic case. (default: ``None``)
    :return:
        The distance to the break position in pc.
    """
    # Parabolic case
    if theta_jet is None:
        return gamma_in*sigma_M*R_L/pc
    # Conical case
    else:
        return np.sqrt(gamma_in*sigma_M)*R_L/theta_jet/pc


if __name__ == "__main__":
    z = 0.00436
    D = 16.8e+06*u.pc
    sigma_D = 0.8e+06*u.pc
    M_BH_sun = 6e+09
    gamma_in = 1.01
    sigma_M = 20
    a = 0.5
    R_L = get_R_L(M_BH_sun*M_sun, a)
    print("R_L = ", R_L)
    theta_jet = np.deg2rad(10.)
    theta_obs = np.deg2rad(17.)
    print("z1 through M,a = ", get_z1_Ma(gamma_in, sigma_M, M_BH_sun, a, theta_jet))
    print("z1 through R_L = ", get_z1_RL(gamma_in, sigma_M, R_L, theta_jet))
    print("Ang2dist = ", ang_to_dist(z))
    l_mas = 1
    print("1 mas corresponds to {} pc of de-projected distance".format(mas_to_pc(l_mas, z, theta_obs)))
    print("1 mas corresponds to {} pc of de-projected distance".format(l_mas*u.mas.to(u.rad)*D/np.sin(theta_obs)))

    # post_samples = np.loadtxt("Release/posterior_sample_each10.txt")
    # z1_proj_mas_samples = np.exp(post_samples[:, 6])
    # fig, axes = plt.subplots(1, 1)
    # axes.hist(z1_proj_mas_samples, bins=30)
    # axes.set_xlabel(r"$z_{\rm 1,proj}$, mas")
    # plt.show()
    #
    # mad = median_abs_deviation(z1_proj_mas_samples)
    # sigma_mad = 1.4826*mad
    # sigma_raw = np.std(z1_proj_mas_samples)
    # r_br = np.median(z1_proj_mas_samples)
    # print(r_br, sigma_raw, sigma_mad)