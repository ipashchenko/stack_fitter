import sys
import os
import dnest4
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import median_abs_deviation
import corner
from astropy import units as u, constants as const
from inference import get_R_g, get_r1_Ma, get_z1_Ma, M_sun
import emcee

rad_to_mas = u.rad.to(u.mas)

stack_post_samples = np.loadtxt("Release/posterior_sample_each10.txt")
z1_proj_mas_samples = np.exp(stack_post_samples[:, 6])
r0_proj_mas_samples = np.exp(stack_post_samples[:, 4])
z1_proj_mas_samples = z1_proj_mas_samples + r0_proj_mas_samples
z1_obs = np.median(z1_proj_mas_samples)
mad = median_abs_deviation(z1_proj_mas_samples)
print(z1_obs)
r1_obs_mas = 2.5
r1_obs_sigma_mas = 0.5
z1_sigma = 1.4826*mad
M_BH_sun = 6.5e+09
M_BH = M_BH_sun*M_sun


def log_prior(params):
    lg_sigma_M, M_BH_sun, D, a = params
    return stats.uniform.logpdf(lg_sigma_M, 0, 6) +\
        stats.norm.logpdf(M_BH_sun, loc=6.5, scale=0.9) +\
        stats.norm.logpdf(D, loc=16.8, scale=0.8) +\
        stats.uniform.logpdf(a, 0, 1)

def log_likelihood(params, y_obs, sigma_y):
    lg_sigma_M, M_BH_sun, D, a = params
    r1_model_pc = get_r1_Ma(1.1, 10**lg_sigma_M, M_BH_sun*1e+09, a)
    # In mas
    r1_model_mas = rad_to_mas*r1_model_pc/(1e+06*D)
    return -0.5*np.log(2*np.pi*sigma_y**2) - 0.5*(y_obs - r1_model_mas)**2/sigma_y**2


def log_prob(params, y_obs, sigma_y):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, y_obs, sigma_y)



pos = np.array([1., 6.5, 16.8, 0.2]) + 1e-2 * np.random.randn(64, 4)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(r1_obs_mas, r1_obs_sigma_mas))
sampler.run_mcmc(pos, 30000, progress=True)

tau = sampler.get_autocorr_time()
print(tau)

sys.exit(0)

flat_samples = sampler.get_chain(discard=100, thin=30, flat=True)
print(flat_samples.shape)
n_params = flat_samples.shape[1]
post_samples = flat_samples
n_minus = 0
post_samples_use = np.vstack((post_samples[:, 0],
                              post_samples[:, 1],
                              post_samples[:, 2],
                              post_samples[:, 3])).T
fig, axes = plt.subplots(nrows=n_params-n_minus, ncols=n_params-n_minus)
fig.set_size_inches(12.5, 12.5)
fig = corner.corner(post_samples_use,
                    # show_titles=True,
                    labels=[r"$\log_{10}{\sigma_{\rm M}}$",
                            r"${M_{\rm BH}}$, [$10^9 M_{\rm sun}$]",
                            r"$D$, [Mpc]",
                            # r"$\theta_{\rm obs}$, $^{\circ}$",
                            # r"$\theta_{\rm jet}$, $^{\circ}$",
                            r"$a$"],
                            # r"$\log_{10}{a}$"],
                    # quantiles=[0.16, 0.50, 0.84],
                    plot_contours=True, color="gray", range=(n_params-n_minus)*[1.0],
                    plot_datapoints=False, fill_contours=True,
                    hist2d_kwargs={"plot_datapoints": False,
                                   "plot_density": True,
                                   "plot_contours": True,
                                   "no_fill_contours": True},
                    hist_kwargs={# 'normed': True,
                        # 'histtype': 'step',
                        # 'stacked': True,
                        'ls': 'solid',
                        'density': True},
                    levels=(0.393, 0.865, 0.989),
                    fig=fig)
fig.savefig("inference_using_r1_all.png", bbox_inches="tight", dpi=300)
plt.show()