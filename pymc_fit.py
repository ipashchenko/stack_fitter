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


# For tics and line widths. Re-write colors and figure size later.
plt.style.use('science')
# Default color scheme
matplotlib.rcParams['axes.prop_cycle'] = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
# Default figure size
matplotlib.rcParams['figure.figsize'] = (6.4, 4.8)

data_file = "/home/ilya/github/stack_fitter/real/mojave/za_rs_0.1max_2g_custom_mojave.txt"
save_dir = "/home/ilya/github/stack_fitter/real/mojave"
# data_file = "/home/ilya/github/stack_fitter/simulations/zs_rs.txt"
# save_dir = "/home/ilya/github/stack_fitter/simulations"

n_each = 1
r, R = np.loadtxt(data_file, unpack=True)
max_r = 15
mask = r < max_r
r = r[mask]
R = R[mask]
# R = medfilt(R, 3)
r_min = np.min(r)
r_max = np.max(r)
r = r[::n_each]
R = R[::n_each]
rnew = np.linspace(np.min(r), np.max(r), len(r))
r = r[:, None]
rnew = rnew[:, None]


class MyMean(pm.gp.mean.Mean):
    def __init__(self, a, b, r0):
        super().__init__()
        self.a = a
        self.b = b
        self.r0 = r0

    def __call__(self, X):
        return pt.alloc(1.0, X.shape[0]) * pt.squeeze(pt.exp(self.b)*pt.power(X + pt.exp(self.r0), self.a))


class MyMeanCP(pm.gp.mean.Mean):
    def __init__(self, a_before, b_before, a_after, r0, r1, cp):
        super().__init__()
        self.a_before = a_before
        self.b_before = b_before
        self.a_after = a_after
        self.b_after = b_before + a_before*pt.log(cp + pt.exp(r0)) - a_after*pt.log(cp + r1)

        self.r0 = r0
        self.r1 = r1
        self.cp = cp

    def __call__(self, X):
        y = pt.switch(X < self.cp,
                      pt.exp(self.b_before)*pt.power(X + pt.exp(self.r0), self.a_before),
                      pt.exp(self.b_after)*pt.power(X + self.r1, self.a_after))
        return pt.alloc(1.0, X.shape[0]) * pt.squeeze(y)


def scaling_function(x, a, b, x0):
    return pt.exp(b)*pt.power(x + pt.exp(x0), a)

########################################################################################################################

with pm.Model() as model:
    a_before = pm.Normal("a_before", mu=1., sigma=0.25)
    b_before = pm.Normal("b_before", mu=0., sigma=1.0)
    a_after = pm.Normal("a_after", mu=1., sigma=0.25)
    # r0 = pm.Normal("r0", mu=-2., sigma=1.0)
    # FIXME: artificial data case
    r0 = pm.Normal("r0", mu=-2., sigma=1.0)
    r1 = pm.Normal("r1", mu=0., sigma=0.5)
    cp = pm.Uniform("cp", lower=r_min, upper=r_max)
    b_after = pm.Deterministic("b_after", b_before + a_before*pt.log(cp + pt.exp(r0)) - a_after*pt.log(cp + r1))
    mean = MyMeanCP(a_before, b_before, a_after, r0, r1, cp)

    eta = pm.HalfNormal("eta", sigma=0.02, initval=0.02)
    # FIXME: artificial data case
    # eta = pm.HalfNormal("eta", sigma=0.2, initval=0.2)
    logalpha = pm.Uniform("logalpha", lower=-5, upper=5)
    sigma = pm.HalfCauchy("sigma", beta=0.1, initval=0.1)
    # l = 1.4
    l = pm.HalfCauchy("l", beta=0.1, initval=0.1)
    # cov = eta**2 * pm.gp.cov.ExpQuad(1, l)
    cov = eta**2 * pm.gp.cov.RatQuad(1, pt.exp(logalpha), l)

    # c = pm.Deterministic("c", -pt.exp(r0))
    # offset = pm.HalfCauchy("offset", beta=0.1, initval=0.1)
    # cov_poly = pm.gp.cov.Polynomial(1, c=c, d=0.5, offset=offset)

    # Multiplication
    # cov_total = cov * cov_poly
    cov_total = cov

    # Scaling covariance
    # cov_total = pm.gp.cov.ScaledCov(1, scaling_func=scaling_function, args=(a, b, r0), cov_func=cov)
    # Add white noise to stabilise
    # cov_total += pm.gp.cov.WhiteNoise(1e-5)


    gp = pm.gp.Marginal(mean_func=mean, cov_func=cov_total)
    y_ = gp.marginal_likelihood("y", X=r, y=R, sigma=sigma)

    mp = pm.find_MAP(include_transformed=True)

    print(sorted([name + ":" + str(mp[name]) for name in mp.keys() if not name.endswith("_")]))


    a_before_mp = float(mp['a_before'])
    b_before_mp = float(mp['b_before'])
    a_after_mp = float(mp['a_after'])
    b_after_mp = float(mp['b_after'])
    r0_mp = float(mp['r0'])
    r1_mp = float(mp['r1'])
    cp_mp = float(mp["cp"])

    mu, var = gp.predict(rnew, point=mp, diag=True)
    only_gp = np.where(rnew[:, 0] < cp_mp,
                       mu - np.exp(b_before_mp)*(rnew[:, 0] + np.exp(r0_mp))**a_before_mp,
                       mu - np.exp(b_after_mp)*(rnew[:, 0] + r1_mp)**a_after_mp)
    model = np.where(rnew[:, 0] < cp_mp,
                     np.exp(b_before_mp)*(rnew[:, 0] + np.exp(r0_mp))**a_before_mp,
                     np.exp(b_after_mp)*(rnew[:, 0] + r1_mp)**a_after_mp)
    fig, axes = plt.subplots(1, 1)
    axes.scatter(r[:, 0], R)
    axes.plot(rnew[:, 0], only_gp, color="C1")
    axes.plot(rnew[:, 0], model, color="red")
    axes.fill_between(rnew[:, 0], only_gp - np.sqrt(var), only_gp + np.sqrt(var), color="C1", alpha=0.5)
    plt.axhline(0.0)
    axes.set_xlabel("r, mas")
    axes.set_ylabel("R, mas")
    axes.text(0.03, 0.90, r"$d_1$ = {:.2f}".format(a_before_mp),
               fontdict={"fontsize": small_font}, transform=axes.transAxes, ha="left")
    axes.text(0.03, 0.85, r"$d_2$ = {:.2f}".format(a_after_mp),
               fontdict={"fontsize": small_font}, transform=axes.transAxes, ha="left")
    axes.text(0.03, 0.80, r"$r_{{\rm break}}$ = {:.2f}".format(cp_mp),
               fontdict={"fontsize": small_font}, transform=axes.transAxes, ha="left")
    axes.axvline(cp_mp, lw=1, color="k", ls="--")
    fig.savefig(os.path.join(save_dir, "pymc_changepoint_0.1max_2g_custom_mojave_up_to_15mas.png"), bbox_inches="tight", dpi=300)
    plt.show()


########################################################################################################################

#
# with pm.Model() as model:
#     a = pm.Normal("a", mu=1., sigma=0.25)
#     b = pm.Normal("b", mu=0., sigma=1.0)
#
#     # r0 = pm.Normal("r0", mu=-2., sigma=1.0)
#     # FIXME: artificial data case
#     r0 = pm.Normal("r0", mu=-10., sigma=1.0)
#
#     mean = MyMean(a, b, r0)
#
#     # eta = pm.HalfNormal("eta", sigma=0.02, initval=0.02)
#     # FIXME: artificial data case
#     eta = pm.HalfNormal("eta", sigma=0.2, initval=0.2)
#
#     logalpha = pm.Uniform("logalpha", lower=-5, upper=5)
#
#     sigma = pm.HalfCauchy("sigma", beta=0.1, initval=0.1)
#
#     # l = 1.0
#     l = pm.HalfCauchy("l", beta=0.1, initval=0.1)
#     # cov = eta**2 * pm.gp.cov.ExpQuad(1, l)
#     cov = eta**2 * pm.gp.cov.RatQuad(1, pt.exp(logalpha), l)
#
#
#     # c = pm.Deterministic("c", -pt.exp(r0))
#     # offset = pm.HalfCauchy("offset", beta=0.1, initval=0.1)
#     # cov_poly = pm.gp.cov.Polynomial(1, c=c, d=a, offset=offset)
#     #
#     # # Multiplication
#     # cov_total = cov * cov_poly
#     #
#     # # Scaling covariance
#     # cov_total = pm.gp.cov.ScaledCov(1, scaling_func=scaling_function, args=(a, b, r0), cov_func=cov)
#     # # Add white noise to stabilise
#     # cov_total += pm.gp.cov.WhiteNoise(1e-5)
#
#     gp = pm.gp.Marginal(mean_func=mean, cov_func=cov)
#     y_ = gp.marginal_likelihood("y", X=r, y=R, sigma=sigma)
#
#     mp = pm.find_MAP(include_transformed=True)
#
#     print(sorted([name + ":" + str(mp[name]) for name in mp.keys() if not name.endswith("_")]))
#
#
#     a_mp = float(mp['a'])
#     b_mp = float(mp['b'])
#     r0_mp = float(mp['r0'])
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
