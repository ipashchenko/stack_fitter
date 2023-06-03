import os
import sys
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import scoreatpercentile
sys.path.insert(0, '/home/ilya/github/dnest4postprocessing')
from postprocess import postprocess

label_size = 14
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
data_file = "/home/ilya/github/stack_fitter/m87_r_fwhm.txt"
save_dir = "/home/ilya/github/stack_fitter"
run_dir = "/home/ilya/github/stack_fitter/Release"
fitted_file = "/home/ilya/github/stack_fitter/Release/posterior_sample.txt"

logZ, _, _ = postprocess(plot=False,
                         sample_file=os.path.join(run_dir, 'sample.txt'),
                         level_file=os.path.join(run_dir, 'levels.txt'),
                         sample_info_file=os.path.join(run_dir, 'sample_info.txt'),
                         post_sample_file=os.path.join(run_dir, "posterior_sample.txt"))

post_samples = np.loadtxt(fitted_file)

r, R = np.loadtxt(data_file, unpack=True)
r_grid = np.linspace(np.min(r), np.max(r), 1000)
fig, axes = plt.subplots(1, 1)
axes.set_xlabel("Separation (mas)")
axes.set_ylabel(r"Width (mas)")
axes.scatter(r, R)

if not changepoint:
    if not GP:
        a_samples, b_samples, r0_samples, abs_error_samples, frac_error_samples = np.split(post_samples, 5, 1)
    else:
        a_samples, b_samples, r0_samples, abs_error_samples, frac_error_samples, gp_amp_samples = np.split(post_samples, 6, 1)

    low, med, up = scoreatpercentile(a_samples, [16, 50, 84])
    low_r0, med_r0, up_r0 = scoreatpercentile(np.exp(r0_samples), [16, 50, 84])
    axes.text(0.03, 0.90, "d = {:.2f} ({:.2f}, {:.2f})".format(med, low, up), fontdict={"fontsize": 10},
              transform=axes.transAxes, ha="left")
    axes.text(0.03, 0.85, r"$r_0$ (mas) = {:.2f} ({:.2f}, {:.2f})".format(med_r0, low_r0, up_r0), fontdict={"fontsize": 10},
              transform=axes.transAxes, ha="left")
    for i in np.random.randint(low=0, high=len(a_samples), size=200):
        a = a_samples[i]
        b = b_samples[i]
        r0 = np.exp(r0_samples[i])
        axes.plot(r_grid, np.exp(b)*(r_grid + r0)**a, color="red", alpha=0.025, zorder=3)

else:
    if not GP:
        a_before_samples, a_after_samples, b_before_samples, b_after_samples, r0_samples, r1_samples,\
            log_changepoint_samples, abs_error_samples, frac_error_samples = np.split(post_samples, 9, 1)
    else:
        a_before_samples, a_after_samples, b_before_samples, b_after_samples, r0_samples, r1_samples, \
            log_changepoint_samples, abs_error_samples, frac_error_samples, gp_amp_samples = np.split(post_samples, 10, 1)
    low_a_before, med_a_before, up_a_before = scoreatpercentile(a_before_samples, [16, 50, 84])
    low_a_after, med_a_after, up_a_after = scoreatpercentile(a_after_samples, [16, 50, 84])
    low_cp, med_cp, up_cp = scoreatpercentile(np.exp(log_changepoint_samples), [16, 50, 84])
    low_r0, med_r0, up_r0 = scoreatpercentile(np.exp(r0_samples), [16, 50, 84])

    axes.text(0.03, 0.90, r"$d_1$ = {:.2f} ({:.2f}, {:.2f})".format(med_a_before, low_a_before, up_a_before),
               fontdict={"fontsize": 10}, transform=axes.transAxes, ha="left")
    axes.text(0.03, 0.85, r"$d_2$ = {:.2f} ({:.2f}, {:.2f})".format(med_a_after, low_a_after, up_a_after),
               fontdict={"fontsize": 10}, transform=axes.transAxes, ha="left")
    axes.text(0.03, 0.80, r"$r_{{\rm break}}$ = {:.2f} ({:.2f}, {:.2f})".format(med_cp, low_cp, up_cp),
               fontdict={"fontsize": 10}, transform=axes.transAxes, ha="left")
    axes.text(0.03, 0.75, r"$r_0$ (mas) = {:.2f} ({:.2f}, {:.2f})".format(med_r0, low_r0, up_r0),
               fontdict={"fontsize": 10}, transform=axes.transAxes, ha="left")

    for i in np.random.randint(low=0, high=len(a_before_samples), size=100):
        log_changepoint = log_changepoint_samples[i]
        r0 = np.exp(r0_samples[i])
        r1 = r1_samples[i]
        after = r_grid >= np.exp(log_changepoint)
        before = r_grid < np.exp(log_changepoint)
        axes.plot(r_grid[after], np.exp(b_after_samples[i])*(r_grid[after] + r1)**a_after_samples[i], color="red", alpha=0.025, zorder=3)
        axes.plot(r_grid[before], np.exp(b_before_samples[i])*(r_grid[before] + r0)**a_before_samples[i], color="red", alpha=0.025, zorder=4)

if logZ is not None:
    axes.text(0.03, 0.95, "logZ = {:.2f}".format(logZ), fontdict={"fontsize": 10}, transform=axes.transAxes, ha="left")

if changepoint:
    fig_name = "fitted_2.png"
else:
    fig_name = "fitted_1.png"

fig.savefig(fig_name, bbox_inches="tight", dpi=300)
plt.show()
