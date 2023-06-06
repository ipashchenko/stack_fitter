import pymc as pm
import numpy as np
import pytensor.tensor as pt
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

data_file = "/home/ilya/github/stack_fitter/m87_r_fwhm.txt"

n_each = 2
r, R = np.loadtxt(data_file, unpack=True)
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



with pm.Model() as model:
    a_before = pm.Normal("a_before", mu=1., sigma=0.5)
    b_before = pm.Normal("b_before", mu=-2., sigma=1.0)
    a_after = pm.Normal("a_after", mu=1., sigma=0.5)
    r0 = pm.Normal("r0", mu=-2., sigma=1.0)
    r1 = pm.Normal("r1", mu=0., sigma=0.5)
    cp = pm.Uniform("cp", lower=r_min, upper=r_max)
    b_after = pm.Deterministic("b_after", b_before + a_before*pt.log(cp + pt.exp(r0)) - a_after*pt.log(cp + r1))
    mean = MyMeanCP(a_before, b_before, a_after, r0, r1, cp)

    eta = pm.HalfNormal("eta", sigma=0.025, initval=0.025)
    sigma = pm.HalfCauchy("sigma", beta=0.1, initval=0.1)
    l = 1.0
    cov = eta**2 * pm.gp.cov.ExpQuad(1, l)

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
    plt.show()




# with pm.Model() as model:
#     a = pm.Normal("a", mu=1., sigma=0.5)
#     b = pm.Normal("b", mu=-2., sigma=1.0)
#     r0 = pm.Normal("r0", mu=-2., sigma=1.0)
#     mean = MyMean(a, b, r0)
#
#     eta = pm.HalfCauchy("eta", beta=2., initval=2.0)
#     sigma = pm.HalfCauchy("sigma", beta=0.1, initval=0.1)
#     l = 1.0
#     cov = eta**2 * pm.gp.cov.ExpQuad(1, l)
#
#     c = pm.Deterministic("c", -pt.exp(r0))
#     offset = pm.HalfCauchy("offset", beta=0.1, initval=0.1)
#     cov_poly = pm.gp.cov.Polynomial(1, c=c, d=a, offset=offset)
#
#     # Multiplication
#     # cov_total = cov * cov_poly
#
#     # Scaling covariance
#     cov_total = pm.gp.cov.ScaledCov(1, scaling_func=scaling_function, args=(a, b, r0), cov_func=cov)
#     # Add white noise to stabilise
#     cov_total += pm.gp.cov.WhiteNoise(1e-5)
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
#     a_mp = float(mp['a'])
#     b_mp = float(mp['b'])
#     r0_mp = float(mp['r0'])
#
#     mu, var = gp.predict(rnew, point=mp, diag=True)
#     only_gp = mu - np.exp(b_mp)*(rnew[:, 0] + np.exp(r0_mp))**a_mp
#
#     fig, axes = plt.subplots(1, 1)
#     axes.scatter(r[:, 0], R)
#     axes.plot(rnew[:, 0], only_gp, color="C1")
#     axes.fill_between(rnew[:, 0], only_gp - np.sqrt(var), only_gp + np.sqrt(var), color="C1", alpha=0.5)
#     plt.axhline(0.0)
#     axes.plot(rnew[:, 0], np.exp(b_mp)*(rnew[:, 0] + np.exp(r0_mp))**a_mp, color="red")
#     axes.set_xlabel("r, mas")
#     axes.set_ylabel("R, mas")
#     plt.show()
