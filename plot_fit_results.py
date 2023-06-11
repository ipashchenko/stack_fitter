import os
import sys
import matplotlib
matplotlib.use("TkAgg")
import scienceplots
import matplotlib.pyplot as plt
# For tics and line widths. Re-write colors and figure size later.
plt.style.use('science')
import numpy as np
from scipy.stats import scoreatpercentile
sys.path.insert(0, '/home/ilya/github/dnest4postprocessing')
from postprocess import postprocess
from gp_utils import make_GP_SE_predictions, make_GP_RQ_predictions, gp_dist_plot

label_size = 18
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['axes.titlesize'] = label_size
matplotlib.rcParams['axes.labelsize'] = label_size
matplotlib.rcParams['font.size'] = label_size
matplotlib.rcParams['legend.fontsize'] = label_size
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


changepoint = False
GP = True
# gp_scale = 1.0
n_pred = 1000
# n_pred = 300
n_each = 2
alpha = 0.05
small_font = 12
data_file = "/home/ilya/github/stack_fitter/m87_r_fwhm.txt"
save_dir = "/home/ilya/github/stack_fitter"
run_dir = "/home/ilya/github/stack_fitter/Release"
fitted_file = "/home/ilya/github/stack_fitter/Release/posterior_sample.txt"

logZ, _, _ = postprocess(plot=True,
                         sample_file=os.path.join(run_dir, 'sample.txt'),
                         level_file=os.path.join(run_dir, 'levels.txt'),
                         sample_info_file=os.path.join(run_dir, 'sample_info.txt'),
                         post_sample_file=os.path.join(run_dir, "posterior_sample.txt"))
# logZ = 1.

post_samples = np.loadtxt(fitted_file)
n_post = len(post_samples)

r, R = np.loadtxt(data_file, unpack=True)
r_grid = np.linspace(np.min(r), np.max(r), 1000)
r = r[::n_each]
R = R[::n_each]
fig, axes = plt.subplots(1, 1, figsize=(6.4, 4.8))
# fig, axes = plt.subplots(1, 1)#, figsize=(9.6, 7.2))
axes.set_xlabel("Separation (mas)")
axes.set_ylabel(r"Width (mas)")
axes.scatter(r, R, zorder=5, color="black", s=3)
axes.set_ylim([-2, 5])

if n_pred > n_post:
    n_pred = n_post
draw_idx = np.random.choice(np.arange(n_post, dtype=int), n_pred, replace=False)

if not changepoint:
    if not GP:
        a_samples, b_samples, r0_samples, abs_error_samples, frac_error_samples = np.split(post_samples, 5, 1)
    else:
        a_samples, b_samples, r0_samples, abs_error_samples, frac_error_samples, gp_logamp_samples, gp_scale_samples,\
            gp_logalpha_samples = np.split(post_samples, 8, 1)

    low, med, up = scoreatpercentile(a_samples, [16, 50, 84])
    low_r0, med_r0, up_r0 = scoreatpercentile(np.exp(r0_samples), [16, 50, 84])
    axes.text(0.03, 0.90, "d = {:.2f} ({:.2f}, {:.2f})".format(med, low, up), fontdict={"fontsize": small_font},
              transform=axes.transAxes, ha="left")
    # axes.text(0.03, 0.85, r"$r_0$ (mas) = {:.2f} ({:.2f}, {:.2f})".format(med_r0, low_r0, up_r0), fontdict={"fontsize": small_font},
    #           transform=axes.transAxes, ha="left")
    axes.axhline(0., lw=2, color="black")

    model_grid_samples = list()
    GP_only_grid_samples = list()

    for i in draw_idx:
        a = a_samples[i]
        b = b_samples[i]
        r0 = np.exp(r0_samples[i])
        mu_model = np.exp(b)*(r + r0)**a
        mu_model_grid = np.exp(b)*(r_grid + r0)**a
        model_grid_samples.append(mu_model_grid)
        if GP:
            frac_error = np.exp(frac_error_samples[i])
            abs_error = np.exp(abs_error_samples[i])
            amp_gp = np.exp(gp_logamp_samples[i])
            scale_gp = gp_scale_samples[i]
            alpha_gp = np.exp(gp_logalpha_samples[i])
            # sigma = np.hypot(mu_model*frac_error, abs_error)
            # FIXME: Here sigma must be a scalar!
            # mu_pred, cov_pred = make_GP_SE_predictions(r_grid, r, R - mu_model, amp_gp, scale_gp, np.sqrt(1e-05))
            mu_pred, cov_pred = make_GP_RQ_predictions(r_grid, r, R - mu_model, amp_gp, scale_gp, alpha_gp, np.sqrt(1e-05))
            # mu_pred, cov_pred = make_GP_SElin_predictions(r_grid, r, R - mu_model, amp_gp, gp_scale, 1e-05, 1, -r0, sigma)
            R_pred = np.random.multivariate_normal(mu_pred, cov_pred)
            GP_only_grid_samples.append(R_pred)
            # axes.plot(r_grid, R_pred, color="C2", alpha=alpha, zorder=4)
        # axes.plot(r_grid, np.exp(b)*(r_grid + r0)**a + R_pred, color="C1", alpha=alpha, zorder=3)
        # axes.plot(r_grid, np.exp(b)*(r_grid + r0)**a, color="C0", alpha=alpha, zorder=3)

    model_grid_samples = np.atleast_2d(model_grid_samples)
    axes = gp_dist_plot(model_grid_samples, r_grid, axes, "Reds")
    if GP:
        GP_only_grid_samples = np.atleast_2d(GP_only_grid_samples)
        axes = gp_dist_plot(GP_only_grid_samples, r_grid, axes, "Blues")

else:
    if not GP:
        a_before_samples, a_after_samples, b_before_samples, b_after_samples, r0_samples, r1_samples,\
            log_changepoint_samples, abs_error_samples, frac_error_samples = np.split(post_samples, 9, 1)
    else:
        a_before_samples, a_after_samples, b_before_samples, b_after_samples, r0_samples, r1_samples, \
            log_changepoint_samples, abs_error_samples, frac_error_samples, gp_logamp_samples, gp_scale_samples,\
            gp_logalpha_samples = np.split(post_samples, 12, 1)
    low_a_before, med_a_before, up_a_before = scoreatpercentile(a_before_samples, [16, 50, 84])
    low_a_after, med_a_after, up_a_after = scoreatpercentile(a_after_samples, [16, 50, 84])
    low_cp, med_cp, up_cp = scoreatpercentile(np.exp(log_changepoint_samples), [16, 50, 84])
    low_r0, med_r0, up_r0 = scoreatpercentile(np.exp(r0_samples), [16, 50, 84])

    axes.text(0.03, 0.90, r"$d_1$ = {:.2f} ({:.2f}, {:.2f})".format(med_a_before, low_a_before, up_a_before),
               fontdict={"fontsize": small_font}, transform=axes.transAxes, ha="left")
    axes.text(0.03, 0.85, r"$d_2$ = {:.2f} ({:.2f}, {:.2f})".format(med_a_after, low_a_after, up_a_after),
               fontdict={"fontsize": small_font}, transform=axes.transAxes, ha="left")
    axes.text(0.03, 0.80, r"$r_{{\rm break}}$ = {:.2f} ({:.2f}, {:.2f})".format(med_cp, low_cp, up_cp),
               fontdict={"fontsize": small_font}, transform=axes.transAxes, ha="left")
    # axes.text(0.03, 0.75, r"$r_0$ (mas) = {:.2f} ({:.2f}, {:.2f})".format(med_r0, low_r0, up_r0),
    #            fontdict={"fontsize": small_font}, transform=axes.transAxes, ha="left")

    model_samples = list()
    model_grid_samples = list()
    GP_only_grid_samples = list()

    for i in draw_idx:
        log_changepoint = log_changepoint_samples[i]
        r0 = np.exp(r0_samples[i])
        r1 = r1_samples[i]
        after_grid = r_grid >= np.exp(log_changepoint)
        before_grid = r_grid < np.exp(log_changepoint)
        after = r >= np.exp(log_changepoint)
        before = r < np.exp(log_changepoint)
        mu_model_after = np.exp(b_after_samples[i])*(r[after] + r1)**a_after_samples[i]
        mu_model_before = np.exp(b_before_samples[i])*(r[before] + r0)**a_before_samples[i]
        mu_model = np.hstack((mu_model_before, mu_model_after))
        model_samples.append(mu_model)

        mu_model_after = np.exp(b_after_samples[i])*(r_grid[after_grid] + r1)**a_after_samples[i]
        mu_model_before = np.exp(b_before_samples[i])*(r_grid[before_grid] + r0)**a_before_samples[i]
        mu_model_grid = np.hstack((mu_model_before, mu_model_after))
        model_grid_samples.append(mu_model_grid)

        if GP:
            frac_error = np.exp(frac_error_samples[i])
            abs_error = np.exp(abs_error_samples[i])
            amp_gp = np.exp(gp_logamp_samples[i])
            scale_gp = gp_scale_samples[i]
            alpha_gp = np.exp(gp_logalpha_samples[i])
            sigma = np.hypot(mu_model*frac_error, abs_error)
            # FIXME: Here sigma must be a scalar!
            # mu_pred, cov_pred = make_GP_SE_predictions(r_grid, r, R - mu_model, amp_gp, gp_scale, sigma)
            # FIXME:
            # mu_pred, cov_pred = make_GP_SE_predictions(r_grid, r, R - mu_model, amp_gp, gp_scale, np.sqrt(1e-05))
            mu_pred, cov_pred = make_GP_RQ_predictions(r_grid, r, R - mu_model, amp_gp, scale_gp, alpha_gp, np.sqrt(1e-05))
            R_pred = np.random.multivariate_normal(mu_pred, cov_pred)
            GP_only_grid_samples.append(R_pred)
            # axes.plot(r_grid, R_pred, color=cmap(0.9), alpha=samples_alpha, zorder=4)
            # axes.plot(r_grid[after_grid], np.exp(b_after_samples[i])*(r_grid[after_grid] + r1)**a_after_samples[i] +
            #           R_pred[after_grid], color="C1", alpha=alpha, zorder=3)
            # axes.plot(r_grid[before_grid], np.exp(b_before_samples[i])*(r_grid[before_grid] + r0)**a_before_samples[i] +
            #           R_pred[before_grid], color="C1", alpha=alpha, zorder=3)


        # axes.plot(r_grid[after_grid], np.exp(b_after_samples[i])*(r_grid[after_grid] + r1)**a_after_samples[i],
        #           color="C0", alpha=alpha, zorder=3)
        # axes.plot(r_grid[before_grid], np.exp(b_before_samples[i])*(r_grid[before_grid] + r0)**a_before_samples[i],
        #           color="C0", alpha=alpha, zorder=3)

    model_grid_samples = np.atleast_2d(model_grid_samples)
    axes = gp_dist_plot(model_grid_samples, r_grid, axes, "Reds")
    if GP:
        GP_only_grid_samples = np.atleast_2d(GP_only_grid_samples)
        axes = gp_dist_plot(GP_only_grid_samples, r_grid, axes, "Blues")

    axes.axhline(0., lw=2, color="black")



if logZ is not None:
    axes.text(0.03, 0.95, "logZ = {:.2f}".format(logZ), fontdict={"fontsize": small_font}, transform=axes.transAxes, ha="left")

if changepoint:
    if GP:
        fig_name = "fitted_gp_2_science.png"
    else:
        fig_name = "fitted_2.png"
else:
    if GP:
        fig_name = "fitted_gp_1_science.png"
    else:
        fig_name = "fitted_1.png"


fig.savefig(fig_name, bbox_inches="tight", dpi=300)
plt.show()
