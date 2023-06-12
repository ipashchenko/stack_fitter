import sys
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from astropy import units as u, constants as const, cosmology
from scipy.stats import median_abs_deviation, scoreatpercentile
import matplotlib.pyplot as plt
from cycler import cycler
import scienceplots
# For tics and line widths. Re-write colors and figure size later.
plt.style.use('science')
# Default color scheme
matplotlib.rcParams['axes.prop_cycle'] = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
# Default figure size
matplotlib.rcParams['figure.figsize'] = (6.4, 4.8)

label_size = 18
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['axes.titlesize'] = label_size
matplotlib.rcParams['axes.labelsize'] = label_size
matplotlib.rcParams['font.size'] = label_size
matplotlib.rcParams['legend.fontsize'] = label_size
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

cosmo = cosmology.WMAP9
rad_to_mas = u.rad.to(u.mas)

# Speed of light [cm / s]
c = 29979245800.0
# Gravitational constant
G = 6.67430e-08
# Parsec [cm]
pc = 3.0856775814671913e+18
# Mass of the Sun [g]
M_sun = 1.98840987e+33


def get_Gamma_from_beta_app_theta(beta_app, theta):
    beta = beta_app/(np.sin(theta) + beta_app*np.cos(theta))
    return 1./np.sqrt(1. - beta**2)

def ang_to_dist(z):
    return cosmo.kpc_proper_per_arcmin(z)


def mas_to_pc(l_mas, z, theta_obs):
    """
    Given some observed length ``l_mas`` [mas] it returns corresponding de-projected length in pc for a given redshift
    ``z`` and jet viewing angle ``theta_obs`` [rad].
    """
    return (l_mas*u.mas*ang_to_dist(z)).to(u.pc).value/np.sin(theta_obs)

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


def get_sigma(Gamma_max, gamma_in):
    """
    B13 from Nokhrina+2022.

    :param Gamma_max:
        The maximal Lorentz factor.
    :param gamma_in:
        Initial Lorentz factor.
    :return:
    """
    sigma_1 = (-4*(1 + gamma_in - 2*Gamma_max) + 4*np.sqrt((1 + gamma_in - 2*Gamma_max)**2 - (1 - gamma_in)**2))/8
    sigma_2 = (-4*(1 + gamma_in - 2*Gamma_max) - 4*np.sqrt((1 + gamma_in - 2*Gamma_max)**2 - (1 - gamma_in)**2))/8
    return sigma_1


def get_a_from_r1(r1_to_rg, gamma_in, sigma):
    return 8*r1_to_rg*np.sqrt(gamma_in*sigma)/(r1_to_rg**2 + 16*gamma_in*sigma)


def get_jet_power(Psi_total, R_L):
    """
    :param Psi_total:
        Total flux [G*cm^2]
    :param R_L:
        Light cylinder radius [cm]
    :return:
        Jet power [erg/s]
    """
    P_psi = (Psi_total/(np.pi*R_L))**2*c/8
    return P_psi


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

    # Value of Rg for M87 in mas
    Rg_in_mas = (get_R_g(6.5e+09*M_sun)*u.cm/(16.8e+06*u.pc)).to(u.dimensionless_unscaled).value*rad_to_mas


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


    # HST-1 speed
    beta_app = np.array([6.0, 5.48, 6.14, 6.02])
    beta_app_err = np.array([0.48, 0.21, 0.58, 1.05])
    beta_app_mean = np.mean(beta_app)
    beta_app_mean_err = np.sqrt(np.sum(beta_app_err**2))/len(beta_app_err)
    theta_deg = 17.
    theta_deg_err = 2.
    N = int(1e5)
    gamma_in = 1.1
    betas_app = np.random.normal(beta_app_mean, beta_app_mean_err, N)
    thetas = np.deg2rad(np.random.normal(theta_deg, theta_deg_err, N))
    # Some Gamma will be nan, because inconsistent apparent speeds and angle.
    Gammas_sample = get_Gamma_from_beta_app_theta(betas_app, thetas)
    mask = np.isnan(Gammas_sample)
    Gammas_sample = np.ma.array(Gammas_sample, mask=mask)
    sigmas_sample = get_sigma(Gammas_sample, gamma_in)
    low5_sigmas, med5_sigmas, up5_sigmas = scoreatpercentile(sigmas_sample.compressed(), [2.5, 50, 97.5])
    low32_sigmas, med32_sigmas, up32_sigmas = scoreatpercentile(sigmas_sample.compressed(), [16, 50, 84])

    # Our work
    r1_obs_mas = 2.5
    r1_obs_sigma_mas = 0.5
    # Hada+2016
    # r1_obs_mas = 0.55
    # r1_obs_sigma_mas = 0.2*r1_obs_mas
    r1_obs_mas_sample = np.random.normal(r1_obs_mas, r1_obs_sigma_mas, N)
    # g
    M_BH = 6.5e+09*M_sun
    sigma_M_BH = 0.9e+09*M_sun
    M_BH_sample = np.random.normal(M_BH, sigma_M_BH, N)
    # pc
    D = 16.8e+06
    sigma_D = 0.8e+06
    D_sample = np.random.normal(D, sigma_D, N)
    # mas
    Rg_mas_sample = (get_R_g(M_BH_sample)*u.cm/(D_sample*u.pc)).to(u.dimensionless_unscaled).value*rad_to_mas
    r1_to_Rg_sample = r1_obs_mas_sample/Rg_mas_sample

    low5, med, up5 = scoreatpercentile(r1_to_Rg_sample, [2.5, 50, 97.5])
    low32, med, up32 = scoreatpercentile(r1_to_Rg_sample, [16, 50, 84])

    from labellines import labelLines
    import matplotlib.ticker as ticker
    sigmas = np.logspace(0, 2.4, 1000)
    fig, axes = plt.subplots(1, 1, figsize=(6.4, 4.8))
    axes.set_xlim([1., 10**2.4])
    # labels = (r"100$r_{\rm g}$", r"Hada+2016", r"300$r_{\rm g}$", r"This work", r"1000$r_{\rm g}$")
    # labels = (r"100$r_{\rm g}$", r"300$r_{\rm g}$", r"1000$r_{\rm g}$")
    # for i, r1_to_rg in enumerate((100, 157, 300, 654, 1000)):
    # for i, r1_to_rg in enumerate((100, 300, 1000)):
    a_up5 = get_a_from_r1(low5, gamma_in, sigmas)
    a_low5 = get_a_from_r1(up5, gamma_in, sigmas)
    a_up32 = get_a_from_r1(low32, gamma_in, sigmas)
    a_low32 = get_a_from_r1(up32, gamma_in, sigmas)
    axes.fill_between(sigmas, a_low5, a_up5, color="C0", alpha=0.25, label=r"$r_1$ this work")
    axes.fill_between(sigmas, a_low32, a_up32, color="C0", alpha=0.25)
    # axes.plot(sigmas, a, label=r"{}".format(labels[i]), lw=3)
    # labelLines(axes.get_lines(), zorder=2.5, fontsize=14, backgroundcolor="none")
    axes.set_xlabel(r"$\sigma_{\rm M}$")
    axes.set_ylabel(r"$a$")
    axes.set_xscale("log", base=10)
    axes.set_yscale("log", base=10)
    axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    plt.legend(loc="upper left")
    fig.savefig("r1_HST-1_constrains_science.pdf", bbox_inches="tight", dpi=300)




    # Hada+2016
    r1_obs_mas = 0.55
    r1_obs_sigma_mas = 0.2*r1_obs_mas
    r1_obs_mas_sample = np.random.normal(r1_obs_mas, r1_obs_sigma_mas, N)
    # g
    M_BH = 6.5e+09*M_sun
    sigma_M_BH = 0.9e+09*M_sun
    M_BH_sample = np.random.normal(M_BH, sigma_M_BH, N)
    # pc
    D = 16.8e+06
    sigma_D = 0.8e+06
    D_sample = np.random.normal(D, sigma_D, N)
    # mas
    Rg_mas_sample = (get_R_g(M_BH_sample)*u.cm/(D_sample*u.pc)).to(u.dimensionless_unscaled).value*rad_to_mas
    r1_to_Rg_sample = r1_obs_mas_sample/Rg_mas_sample

    low5, med, up5 = scoreatpercentile(r1_to_Rg_sample, [2.5, 50, 97.5])
    low32, med, up32 = scoreatpercentile(r1_to_Rg_sample, [16, 50, 84])

    a_up5 = get_a_from_r1(low5, gamma_in, sigmas)
    a_low5 = get_a_from_r1(up5, gamma_in, sigmas)
    a_up32 = get_a_from_r1(low32, gamma_in, sigmas)
    a_low32 = get_a_from_r1(up32, gamma_in, sigmas)
    axes.fill_between(sigmas, a_low5, a_up5, color="C2", alpha=0.25, label=r"$r_1$ Hada+2016")
    axes.fill_between(sigmas, a_low32, a_up32, color="C2", alpha=0.25)
    axes.axvspan(low5_sigmas, up5_sigmas, color="C1", alpha=0.25, label="HST-1 speeds")
    axes.axvspan(low32_sigmas, up32_sigmas, color="C1", alpha=0.25)
    plt.legend(loc="upper left", prop={"size":12})
    fig.savefig("r1_HST-1_constrains_both_science.pdf", bbox_inches="tight", dpi=300)

    plt.show()

    # sys.exit(0)


    from labellines import labelLines
    import matplotlib.ticker as ticker
    gamma_in = 1.1
    # sigmas = np.logspace(0, 2.302, 100)
    fig, axes = plt.subplots(1, 1, figsize=(6.4, 4.8))
    axes.set_xlim([1., 10**2.4])
    # labels = (r"100$r_{\rm g}$", r"Hada+2016", r"300$r_{\rm g}$", r"This work", r"1000$r_{\rm g}$")
    labels = (r"100$r_{\rm g}$", r"300$r_{\rm g}$", r"1000$r_{\rm g}$")
    colors = ("C0", "C1", "C2")
    # for i, r1_to_rg in enumerate((100, 157, 300, 654, 1000)):
    for i, r1_to_rg in enumerate((100, 300, 1000)):
        a = get_a_from_r1(r1_to_rg, gamma_in, sigmas)
        axes.plot(sigmas, a, label=r"{}".format(labels[i]), lw=3, color=colors[i])
    labelLines(axes.get_lines(), zorder=2.5, fontsize=14, backgroundcolor="none")
    axes.set_xlabel(r"$\sigma_{\rm M}$")
    axes.set_ylabel(r"$a$")
    axes.set_xscale("log", base=10)
    axes.set_yscale("log", base=10)
    axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    fig.savefig("r1_constrain_analytical_science.pdf", bbox_inches="tight", dpi=300)
    plt.show()
