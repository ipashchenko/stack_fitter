import os
import sys
import numpy as np
from astropy import units as u
from astropy.convolution import convolve, Gaussian2DKernel
from scipy.ndimage import rotate
from astropy.stats import mad_std
from astropy.modeling import fitting
from astropy.modeling.models import custom_model, Gaussian1D
from cycler import cycler
import scienceplots
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from from_fits import create_clean_image_from_fits_file


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


def FWHM_ell_beam_slice(bmin, bmaj, PA_diff_bpa):
    """
    FWHM of the elliptical beam slice.

    :param bmin:
        Minor FWHM.
    :param bmaj:
        Major FWHM.
    :param PA_diff_bpa:
        Difference between beam BPA (i.e. PA of the beam major axis) and slice PA. [-np.pi/2, np.pi/2], [rad].
    """
    return bmaj*bmin*np.sqrt((1+np.tan(PA_diff_bpa)**2)/(bmin**2+bmaj**2*np.tan(PA_diff_bpa)**2))


def get_slices(image, pixsize_mas, beam_size_mas, z_obs_min_mas=0.5, z_obs_max_mas=10.0, maxwidth=100, save_dir=None,
               save_prefix="profile", fig=None, alpha=1.0, n_good_min=10):
    if save_dir is None:
        save_dir = os.getcwd()
    std = mad_std(image)
    halfwidth = int(maxwidth/2)
    imsize = image.shape[0]
    halfsize = int(imsize/2)
    nslices = int(z_obs_max_mas/pixsize_mas)
    delta = 1
    widths_mas = list()
    pos_mas = list()
    for i in range(nslices):
        imslice = image[halfsize-halfwidth: halfsize+halfwidth, halfsize+delta*i]
        g_init = Gaussian1D(amplitude=np.max(imslice), mean=halfwidth, stddev=beam_size_mas/pixsize_mas,
                            fixed={'mean': True})
        fit_g = fitting.LevMarLSQFitter()
        x = np.arange(maxwidth)
        y = imslice
        mask = imslice > 5*std
        n_good = np.count_nonzero(mask)
        print("Number of unmasked elements for z = {:.2f} is N = {}".format(delta*i*pixsize_mas, n_good))
        if n_good < n_good_min:
            continue
        g = fit_g(g_init, x[mask], y[mask], weights=1/std)
        print("Convolved FWHM = {:.2f} mas".format(g.fwhm*pixsize_mas))
        width_mas_deconvolved = np.sqrt((g.fwhm*pixsize_mas)**2 - beam_size_mas**2)
        print("Deconvolved FWHM = {:.2f} mas".format(width_mas_deconvolved))
        if np.isnan(width_mas_deconvolved):
            continue
        widths_mas.append(width_mas_deconvolved)
        pos_mas.append(delta*i*pixsize_mas)

    pos_mas = np.array(pos_mas)
    widths_mas = np.array(widths_mas)
    if fig is None:
        fig, axes = plt.subplots(1, 1)
    else:
        axes = fig.get_axes()[0]
    if z_obs_min_mas is not None or z_obs_max_mas is not None:
        assert z_obs_max_mas > z_obs_min_mas
        axes.set_xlim([z_obs_min_mas, z_obs_max_mas])
        mask = np.logical_and(pos_mas < z_obs_max_mas, pos_mas > z_obs_min_mas)
        pos_to_plot = pos_mas[mask]
        widths_to_plot = widths_mas[mask]
    else:
        widths_to_plot = widths_mas
        pos_to_plot = pos_mas

    axes.plot(pos_to_plot, widths_to_plot, color="C0", alpha=alpha)
    axes.set_xlabel(r"$z_{\rm obs}$, mas")
    axes.set_ylabel("FWHM, mas")
    plt.xscale("log")
    plt.yscale("log")
    if save_prefix is not None:
        fig.savefig(os.path.join(save_dir, "{}.png".format(save_prefix)), bbox_inches="tight", dpi=300)
    plt.show()

    return pos_to_plot, widths_to_plot


def fit_profile(pos_to_plot, widths_to_plot, save_prefix="profile_fit", save_dir=None, fig=None, fix_r0=False):
    if save_dir is None:
        save_dir = os.getcwd()
    @custom_model
    def power_law(r, amp=1.0, r0=0.0, k=0.5):
        return amp*(r + r0)**k

    if not fix_r0:
        pl_init = power_law(fixed={"r0": True})
    else:
        pl_init = power_law()
    fit_pl = fitting.LevMarLSQFitter()
    pl = fit_pl(pl_init, pos_to_plot, widths_to_plot, maxiter=10000)
    print(fit_pl.fit_info)
    print("k = ", pl.k)
    print("r0 = ", pl.r0)
    print("amp = ", pl.amp)

    # Plot fit
    xx = np.linspace(np.min(pos_to_plot), np.max(pos_to_plot), 1000)
    yy = pl(xx)
    if fig is None:
        fig, axes = plt.subplots(2, 1, sharex=True, height_ratios=[3, 1])
    else:
        axes = fig.get_axes()[0]
    axes[0].plot(xx, yy, color="C1", label="k = {:.2f}".format(pl.k.value))
    axes[0].scatter(pos_to_plot, widths_to_plot, color="C0", label="data", s=2)
    axes[0].legend()
    axes[0].set_ylabel("FWHM, mas")
    # axes[0].set_xscale("log", base=10.)
    axes[0].set_yscale("log", base=10.)

    # remove the minor ticks
    axes[0].yaxis.set_minor_formatter(ticker.NullFormatter())

    # axes[0].yaxis.set_minor_formatter(ScalarFormatter())
    axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))


    # Only some minor ticks
    # subs = [1.0, 3.0, 5.0, 7.0, 9.0]
    # axes[0].yaxis.set_minor_locator(ticker.LogLocator(subs=subs))

    # Make residuals and plot them
    res = widths_to_plot - pl(pos_to_plot)
    max_res = 1.2*np.max(np.abs(res))

    axes[1].axhline(0, color="k", ls="--", lw=1)
    axes[1].plot(pos_to_plot, res, color="C0", alpha=1.0)
    axes[1].set_ylim([-max_res, max_res])
    # axes.set_ylim([-2, 2])
    axes[1].set_xlabel(r"$z_{\rm obs}$, mas")
    axes[1].set_ylabel("residuals, mas")
    axes[1].set_xscale("log", base=10.)
    # axes[1].xaxis.set_major_formatter(ScalarFormatter())
    axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    # axes[1].xaxis.set_minor_formatter(ticker.FuncFormatter(lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    fig.subplots_adjust(hspace=0)

    if save_prefix is not None:
        fig.savefig(os.path.join(save_dir, "{}.png".format(save_prefix)), bbox_inches="tight", dpi=300)
    plt.show()

    return fig


if __name__ == "__main__":
    save_dir = "/home/ilya/github/stack_fitter/simulations"
    image = np.loadtxt(os.path.join(save_dir, "stack_i.txt"))
    zs, Rs = get_slices(image, pixsize_mas=0.1, beam_size_mas=1.35, save_dir=save_dir,
                        z_obs_min_mas=0.5, z_obs_max_mas=10.)
    fit_profile(zs, Rs, save_dir=save_dir, fix_r0=True)
