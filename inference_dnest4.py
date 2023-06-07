import sys
import os
import dnest4
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import median_abs_deviation
import corner
from astropy import units as u, constants as const
from inference import get_R_g, get_z1_Ma, M_sun
sys.path.insert(0, '/home/ilya/github/dnest4postprocessing')
from postprocess import postprocess


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

class Model(object):

    def __init__(self):
        self.rad_to_mas = u.rad.to(u.mas)

    def from_prior(self):
        lg_sigma_M = np.random.normal(1, 0.5)
        # lg_R_L_rg = np.random.normal(3, 2)
        M_BH_sun = np.random.normal(6.5, 0.9)
        D_Mpc = np.random.normal(16.8, 0.8)
        theta_obs_deg = np.random.normal(17, 2)
        theta_jet_deg = np.random.uniform(5, 15)
        a = np.random.uniform(0, 1)
        return np.array([lg_sigma_M, M_BH_sun, D_Mpc, theta_obs_deg, theta_jet_deg, a])

    def perturb(self, params):
        """
        Unlike in C++, this takes a numpy array of parameters as input,
        and modifies it in-place. The return value is still logH.
        """
        logH = 0.0
        which = np.random.randint(6)

        # lg_sigma_M
        if which == 0:
            logH -= -0.5*((params[which] - 1.)/0.5)**2
            params[which] += 0.5*self.randh()
            logH += -0.5*((params[which] - 1.)/0.5)**2
        # M_BH_sun
        elif which == 1:
            logH -= -0.5*((params[which] - 6.5)/0.9)**2
            params[which] += 0.9*self.randh()
            logH += -0.5*((params[which] - 6.5)/0.9)**2
        # D
        elif which == 2:
            logH -= -0.5*((params[which] - 16.8)/0.8)**2
            params[which] += 0.8*self.randh()
            logH += -0.5*((params[which] - 16.8)/0.8)**2
        # theta_obs_deg
        elif which == 3:
            logH -= -0.5*((params[which] - 17.)/2.)**2
            params[which] += 2.*self.randh()
            logH += -0.5*((params[which] - 17.)/2.)**2
        elif which == 4:
            theta_jet_deg = params[which]
            theta_jet_deg += 10.0*self.randh()
            theta_jet_deg = self.wrap(theta_jet_deg, 5, 15)
            params[which] = theta_jet_deg
        else:
            a = params[which]
            a += 1.0*self.randh()
            a = self.wrap(a, 0, 1)
            params[which] = a

        return logH

    def log_likelihood(self, params):
        """
        Gaussian sampling distribution.
        """
        lg_sigma_M, M_BH_sun, D, theta_obs_deg, theta_jet_deg, a = params
        # In pc
        z1_model_pc = get_z1_Ma(1.1, 10**lg_sigma_M, M_BH_sun*1e+09, a, np.deg2rad(theta_jet_deg))
        # In mas
        z1_model_mas = self.rad_to_mas*(z1_model_pc*np.sin(np.deg2rad(theta_obs_deg))/(1e+06*D))
        # return -0.5*data.shape[0]*np.log(2*np.pi*var) \
        #     - 0.5*np.sum((data[:,1] - (m*data[:,0] + b))**2)/var
        return -0.5*np.log(2*np.pi*z1_sigma**2) - 0.5*(z1_obs - z1_model_mas)**2/z1_sigma/z1_sigma

    def randh(self):
        """
        Generate from the heavy-tailed distribution.
        """
        a = np.random.randn()
        b = np.random.rand()
        t = a/np.sqrt(-np.log(b))
        n = np.random.randn()
        return 10.0**(1.5 - 3*np.abs(t))*n

    def wrap(self, x, a, b):
        assert b > a
        return (x - a)%(b - a) + a


# Create a model object and a sampler
model = Model()
sampler = dnest4.DNest4Sampler(model,
                               backend=dnest4.backends.CSVBackend(".",
                                                                  sep=" "))

# Set up the sampler. The first argument is max_num_levels
gen = sampler.sample(max_num_levels=15, num_steps=10000, new_level_interval=10000,
                     num_per_step=10000, thread_steps=100,
                     num_particles=5, lam=10, beta=100, seed=1234)

# Do the sampling (one iteration here = one particle save)
for i, sample in enumerate(gen):
    print("# Saved {k} particles.".format(k=(i+1)))

fitted_file = "/home/ilya/github/stack_fitter/posterior_sample.txt"
run_dir = "/home/ilya/github/stack_fitter"
logZ, _, _ = postprocess(plot=True,
                         sample_file=os.path.join(run_dir, 'sample.txt'),
                         level_file=os.path.join(run_dir, 'levels.txt'),
                         sample_info_file=os.path.join(run_dir, 'sample_info.txt'),
                         post_sample_file=os.path.join(run_dir, "posterior_sample.txt"))

post_samples = np.loadtxt(fitted_file)
n_post = len(post_samples)
n_params = post_samples.shape[1]
post_samples_use = np.vstack((post_samples[:, 0], post_samples[:, 1], post_samples[:, 2], post_samples[:, 3],
                              post_samples[:, 4], np.log10(post_samples[:, 5]))).T
fig, axes = plt.subplots(nrows=n_params, ncols=n_params)
fig.set_size_inches(15, 15)
fig = corner.corner(post_samples_use,
                    # show_titles=True,
                    labels=[r"$\log_{10}{\sigma_{\rm M}}$",
                            r"${M_{\rm BH}}$, [$10^9 M_{\rm sun}$]",
                            r"$D$, [Mpc]",
                            r"$\theta_{\rm obs}$, $^{\circ}$",
                            r"$\theta_{\rm jet}$, $^{\circ}$",
                            r"$\log_{10}{a}$"],
                    # quantiles=[0.16, 0.50, 0.84],
                    plot_contours=True, color="gray", range=n_params*[1.0],
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
fig.savefig("inference_plus_theta_jet.png", bbox_inches="tight", dpi=300)
plt.show()
