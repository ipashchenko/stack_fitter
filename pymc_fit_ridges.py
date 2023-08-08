import os
import pymc as pm
import numpy as np
import pytensor.tensor as pt
from scipy.signal import medfilt
import matplotlib
matplotlib.use("TkAgg")
import scienceplots
from cycler import cycler
import matplotlib.pyplot as plt
label_size = 18
small_font = 12
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['axes.titlesize'] = label_size
matplotlib.rcParams['axes.labelsize'] = label_size
matplotlib.rcParams['font.size'] = label_size
matplotlib.rcParams['legend.fontsize'] = label_size
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
gaussian_sigma_to_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
gaussian_fwhm_to_sigma = 1.0 / gaussian_sigma_to_fwhm

# For tics and line widths. Re-write colors and figure size later.
plt.style.use('science')
# Default color scheme
matplotlib.rcParams['axes.prop_cycle'] = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
# Default figure size
matplotlib.rcParams['figure.figsize'] = (6.4, 4.8)

# data_file = "/home/ilya/github/stack_fitter/real/zs_rs_real.txt"
# save_dir = "/home/ilya/github/stack_fitter/real"
data_file = "/home/ilya/github/stack_fitter/simulations/zs_rs.txt"
save_dir = "/home/ilya/github/stack_fitter/simulations"

positions = np.loadtxt(os.path.join(save_dir, "positions.dat"))
slices = np.loadtxt(os.path.join(save_dir, "slices.dat"))
slice_x = np.arange(100, dtype=float)
halfwidth = 50
# slice_x = slice_x[:, None]


# ########################################################################################################################
#
# with pm.Model() as model:
#     a_before = pm.Normal("a_before", mu=1., sigma=0.25)
#     b_before = pm.Normal("b_before", mu=0., sigma=1.0)
#     a_after = pm.Normal("a_after", mu=1., sigma=0.25)
#     # r0 = pm.Normal("r0", mu=-2., sigma=1.0)
#     # FIXME: artificial data case
#     r0 = pm.Normal("r0", mu=-2., sigma=1.0)
#     r1 = pm.Normal("r1", mu=0., sigma=0.5)
#     cp = pm.Uniform("cp", lower=r_min, upper=r_max)
#     b_after = pm.Deterministic("b_after", b_before + a_before*pt.log(cp + pt.exp(r0)) - a_after*pt.log(cp + r1))
#     mean = MyMeanCP(a_before, b_before, a_after, r0, r1, cp)
#
#     # eta = pm.HalfNormal("eta", sigma=0.02, initval=0.02)
#     # FIXME: artificial data case
#     eta = pm.HalfNormal("eta", sigma=0.2, initval=0.2)
#     logalpha = pm.Uniform("logalpha", lower=-5, upper=5)
#     sigma = pm.HalfCauchy("sigma", beta=0.1, initval=0.1)
#     # l = 1.0
#     l = pm.HalfCauchy("l", beta=0.1, initval=0.1)
#     # cov = eta**2 * pm.gp.cov.ExpQuad(1, l)
#     cov = eta**2 * pm.gp.cov.RatQuad(1, pt.exp(logalpha), l)
#
#     # c = pm.Deterministic("c", -pt.exp(r0))
#     # offset = pm.HalfCauchy("offset", beta=0.1, initval=0.1)
#     # cov_poly = pm.gp.cov.Polynomial(1, c=c, d=0.5, offset=offset)
#
#     # Multiplication
#     # cov_total = cov * cov_poly
#     cov_total = cov
#
#     # Scaling covariance
#     # cov_total = pm.gp.cov.ScaledCov(1, scaling_func=scaling_function, args=(a, b, r0), cov_func=cov)
#     # Add white noise to stabilise
#     # cov_total += pm.gp.cov.WhiteNoise(1e-5)
#
#
#     gp = pm.gp.Marginal(mean_func=mean, cov_func=cov_total)
#     y_ = gp.marginal_likelihood("y", X=r, y=R, sigma=sigma)
#
#     mp = pm.find_MAP(include_transformed=True)
#
#     print(sorted([name + ":" + str(mp[name]) for name in mp.keys() if not name.endswith("_")]))
#
#
#     a_before_mp = float(mp['a_before'])
#     b_before_mp = float(mp['b_before'])
#     a_after_mp = float(mp['a_after'])
#     b_after_mp = float(mp['b_after'])
#     r0_mp = float(mp['r0'])
#     r1_mp = float(mp['r1'])
#     cp_mp = float(mp["cp"])
#
#     mu, var = gp.predict(rnew, point=mp, diag=True)
#     only_gp = np.where(rnew[:, 0] < cp_mp,
#                        mu - np.exp(b_before_mp)*(rnew[:, 0] + np.exp(r0_mp))**a_before_mp,
#                        mu - np.exp(b_after_mp)*(rnew[:, 0] + r1_mp)**a_after_mp)
#     model = np.where(rnew[:, 0] < cp_mp,
#                      np.exp(b_before_mp)*(rnew[:, 0] + np.exp(r0_mp))**a_before_mp,
#                      np.exp(b_after_mp)*(rnew[:, 0] + r1_mp)**a_after_mp)
#     fig, axes = plt.subplots(1, 1)
#     axes.scatter(r[:, 0], R)
#     axes.plot(rnew[:, 0], only_gp, color="C1")
#     axes.plot(rnew[:, 0], model, color="red")
#     axes.fill_between(rnew[:, 0], only_gp - np.sqrt(var), only_gp + np.sqrt(var), color="C1", alpha=0.5)
#     plt.axhline(0.0)
#     axes.set_xlabel("r, mas")
#     axes.set_ylabel("R, mas")
#     axes.text(0.03, 0.90, r"$d_1$ = {:.2f}".format(a_before_mp),
#                fontdict={"fontsize": small_font}, transform=axes.transAxes, ha="left")
#     axes.text(0.03, 0.85, r"$d_2$ = {:.2f}".format(a_after_mp),
#                fontdict={"fontsize": small_font}, transform=axes.transAxes, ha="left")
#     axes.text(0.03, 0.80, r"$r_{{\rm break}}$ = {:.2f}".format(cp_mp),
#                fontdict={"fontsize": small_font}, transform=axes.transAxes, ha="left")
#     axes.axvline(cp_mp, lw=1, color="k", ls="--")
#     fig.savefig(os.path.join(save_dir, "pymc_changepoint.png"), bbox_inches="tight", dpi=300)
#     plt.show()


########################################################################################################################

with pm.Model() as model:
    a = pm.Normal("a", mu=0.5, sigma=0.5)
    b = pm.Normal("b", mu=0., sigma=1.0)
    r0 = pm.Normal("r0", mu=-10., sigma=1.0)
    logsigma = pm.Normal(f"logsigma", mu=-5, sigma=3.0)

    # Distance of the ridge from the jet center
    ridge_pos = pm.Deterministic("ridge_pos", pt.exp(b)*pt.power(positions + pt.exp(r0), a))
    width_fraction = pm.Uniform("width_fraction", lower=0, upper=0.9)
    width = pm.Deterministic("width", pt.sqrt(width_fraction**2*ridge_pos**2 + (8.5*gaussian_fwhm_to_sigma)**2))

    for i in range(len(positions)):
        # nu = pm.Uniform(f"nu_{i}", lower=2, upper=100)
        logamp = pm.Normal(f"logamp_{i}", mu=-2.0, sigma=3.0)
        slice = pm.Deterministic(f"slice_{i}", pt.exp(logamp)*pt.exp(-((slice_x - halfwidth + ridge_pos[i])/(2*width[i]))**2) +
                                 pt.exp(logamp)*pt.exp(-((slice_x - halfwidth - ridge_pos[i])/(2*width[i]))**2))
        # obs_slice = pm.StudentT(f"obs_slice_{i}", mu=slice, sigma=pt.exp(logsigma), nu=nu, observed=slices[i])
        obs_slice = pm.Normal(f"obs_slice_{i}", mu=slice, sigma=pt.exp(logsigma), observed=slices[i])

    mp = pm.find_MAP(include_transformed=True)




#
#
#
#
#
# with pm.Model() as model:
#     a = pm.Normal("a", mu=0.5, sigma=0.5)
#     b = pm.Normal("b", mu=0., sigma=1.0)
#     r0 = pm.Normal("r0", mu=-10., sigma=1.0)
#     logsigma = pm.Normal(f"logsigma", mu=-5, sigma=3.0)
#
#     # Distance of the ridge from the jet center
#     ridge_pos = pm.Deterministic("ridge_pos", pt.exp(b)*pt.power(positions + pt.exp(r0), a))
#     width_fraction = pm.Uniform("width_fraction", lower=0, upper=0.3)
#     width = pm.Deterministic("width", pt.sqrt(width_fraction**2*ridge_pos**2 + (8.5*gaussian_fwhm_to_sigma)**2))
#
#     for i in range(len(positions)):
#         logamp = pm.Normal(f"logamp_{i}", mu=-2.0, sigma=3.0)
#         slice = pm.Deterministic(f"slice_{i}", pt.exp(logamp)*pt.exp(-((slice_x - halfwidth + ridge_pos[i])/(2*width[i]))**2) +
#                                                pt.exp(logamp)*pt.exp(-((slice_x - halfwidth - ridge_pos[i])/(2*width[i]))**2))
#         obs_slice = pm.StudentT(f"obs_slice_{i}", mu=slice, sigma=pt.exp(logsigma), nu=2.0, observed=slices[i])
#
#     mp = pm.find_MAP(include_transformed=True)


    # print(sorted([name + ":" + str(mp[name]) for name in mp.keys() if not name.endswith("_")]))
#
#
    a_mp = float(mp['a'])
    b_mp = float(mp['b'])
    r0_mp = float(mp['r0'])
#
#     mu, var = gp.predict(rnew, point=mp, diag=True)
#     only_gp = mu - np.exp(b_mp)*(rnew[:, 0] + np.exp(r0_mp))**a_mp
#
#     # with plt.style.context(['science', 'high-contrast']):
#     fig, axes = plt.subplots(1, 1, figsize=(6.4, 4.8))
#     axes.scatter(r[:, 0], R)
#     axes.plot(rnew[:, 0], only_gp, color="C1")
#     axes.fill_between(rnew[:, 0], only_gp - np.sqrt(var), only_gp + np.sqrt(var), color="C1", alpha=0.5)
#     plt.axhline(0.0)
#     axes.plot(rnew[:, 0], np.exp(b_mp)*(rnew[:, 0] + np.exp(r0_mp))**a_mp, color="red")
#     axes.set_xlabel("r, mas")
#     axes.set_ylabel("R, mas")
#     plt.show()
