import os
import sys
import numpy as np
import dlib
from astropy import units as u
from astropy.convolution import convolve, Gaussian2DKernel
from scipy.ndimage import rotate
from astropy.stats import mad_std
import astropy.io.fits as pf
from astropy.modeling import fitting
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.models import custom_model, Gaussian1D
import pybobyqa
from scipy.ndimage import rotate
from cycler import cycler
import matplotlib
import scienceplots
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from spydiff import CLEAN_difmap, CCFITS_to_difmap, convert_difmap_model_file_to_CCFITS, find_image_std, find_bbox
from image import plot as iplot
from from_fits import create_clean_image_from_fits_file, create_image_from_fits_file
matplotlib.use("Agg")

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


gaussian_sigma_to_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
gaussian_fwhm_to_sigma = 1.0 / gaussian_sigma_to_fwhm
gaussian_sigma_to_fwqm = 2.0 * np.sqrt(2.0 * np.log(4.0))
gaussian_sigma_to_fwdm = 2.0 * np.sqrt(2.0 * np.log(10.0))
gaussian_fwhm_to_fwqm = gaussian_sigma_to_fwqm/gaussian_sigma_to_fwhm
gaussian_fwhm_to_fwdm = gaussian_sigma_to_fwdm/gaussian_sigma_to_fwhm


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
               save_prefix="profile", fig=None, alpha=1.0, n_good_min=10, rotation_angle_deg=None, plot=True,
               dlib_max=10.):
    """
    param rotation_angle_deg: (optional)
        Angle (deg) to rotate an image to align the jet with negative RA axis. ``17`` for M87.
        If ``None`` then do not rotate. (default: ``None``)
    """
    if save_dir is None:
        save_dir = os.getcwd()
    # FIXME: Poor estiate for stack I use
    std = mad_std(image)
    # For deep images
    # std = 0.5*6e-05
    positions = list()
    slices = list()
    halfwidth = int(maxwidth/2)
    imsize = image.shape[0]
    halfsize = int(imsize/2)
    nslices = int((z_obs_max_mas-z_obs_min_mas)/pixsize_mas)
    delta = 1
    widths_mas = list()
    pos_mas = list()
    if rotation_angle_deg is not None:
        image = rotate(image, rotation_angle_deg, reshape=False)
    x0 = None
    for i in range(nslices):
        imslice = image[halfsize-halfwidth: halfsize+halfwidth, int(z_obs_min_mas/pixsize_mas) + halfsize+delta*i]
        x = np.arange(maxwidth)
        y = imslice
        mask = imslice > 3*std
        n_good = np.count_nonzero(mask)
        print("Number of unmasked elements for z = {:.2f} is N = {}".format(z_obs_min_mas + delta*i*pixsize_mas, n_good))
        if n_good < n_good_min:
            continue

        # x = x[mask]
        # y = y[mask]

        # width_pixels = fit_single_slice_Gauss1(x, y, halfwidth=halfwidth, beam_size_pixels=beam_size_mas/pixsize_mas, std=std, fixed_mean=True)
        # width_pixels = fit_single_slice_Gauss2(x, y, halfwidth=halfwidth, beam_size_pixels=beam_size_mas/pixsize_mas, std=std)
        # width_pixels = fit_single_slice_Gauss3(x, y, halfwidth=halfwidth, beam_size_pixels=beam_size_mas/pixsize_mas, std=std)
        if z_obs_min_mas + delta*i*pixsize_mas < dlib_max:
            use = "dlib"
        else:
            use = "bobyqa"
        if z_obs_min_mas + delta*i*pixsize_mas < 5.0:
            min_amp_coeff = 0.05
        else:
            min_amp_coeff = 0.05

        width_pixels, x0 = fit_single_slice_Gauss4(x, y, halfwidth=halfwidth, beam_size_pixels=beam_size_mas/pixsize_mas, std=std, plot=plot,
                                                   use=use, save_dir=save_dir, z_obs_mas=z_obs_min_mas + delta*i*pixsize_mas,
                                                   min_amp_coeff=min_amp_coeff, x0=x0)
        slices.append(y)
        positions.append(z_obs_min_mas + delta*i*pixsize_mas)

        print("Convolved width = {:.2f} mas".format(width_pixels*pixsize_mas))

        width_mas_deconvolved = np.sqrt((width_pixels*pixsize_mas)**2 - (beam_size_mas*gaussian_fwhm_to_fwdm)**2)
        print("Deconvolved width = {:.2f} mas".format(width_mas_deconvolved))
        if np.isnan(width_mas_deconvolved):
            continue
        widths_mas.append(width_mas_deconvolved)
        pos_mas.append(z_obs_min_mas + delta*i*pixsize_mas)

    pos_mas = np.array(pos_mas)
    widths_mas = np.array(widths_mas)
    # if fig is None:
    #     fig, axes = plt.subplots(1, 1)
    # else:
    #     axes = fig.get_axes()[0]
    if z_obs_min_mas is not None or z_obs_max_mas is not None:
        assert z_obs_max_mas > z_obs_min_mas
        # axes.set_xlim([z_obs_min_mas, z_obs_max_mas])
        mask = np.logical_and(pos_mas <= z_obs_max_mas, pos_mas >= z_obs_min_mas)
        pos_to_plot = pos_mas[mask]
        widths_to_plot = widths_mas[mask]
    else:
        widths_to_plot = widths_mas
        pos_to_plot = pos_mas

    # axes.plot(pos_to_plot, widths_to_plot, color="C0", alpha=alpha)
    # axes.set_xlabel(r"$z_{\rm obs}$, mas")
    # axes.set_ylabel("FWQM, mas")
    # plt.xscale("log")
    # plt.yscale("log")
    # if save_prefix is not None:
    #     fig.savefig(os.path.join(save_dir, "{}.png".format(save_prefix)), bbox_inches="tight", dpi=300)
    # plt.show()

    return pos_to_plot, widths_to_plot, positions, slices


def fit_single_slice_Gauss1(x, y, **kwargs):
    """
    Fit slice with a single Gaussian.

    :param x:
        Array with pixel numbers.
    :param y:
        Array with image values.
    :param halfwidth:
        Pixel number of the middle of the the slice.

    :return:
        The width in pixels.

    """
    g_init = Gaussian1D(amplitude=np.ma.max(y), mean=kwargs["halfwidth"], stddev=kwargs["beam_size_pixels"],
                        fixed={'mean': kwargs["fixed_mean"]})
    fit_g = fitting.LevMarLSQFitter()
    std = kwargs["std"]
    g = fit_g(g_init, x, y, weights=1/std)
    return g.fwhm


def fit_single_slice_Gauss2(x, y, **kwargs):

    do_plot = kwargs["plot"]
    beam_size_pixels_stddev = gaussian_fwhm_to_sigma*kwargs["beam_size_pixels"]
    std = kwargs["std"]
    min_amp_coeff = kwargs["min_amp_coeff"]
    print("std = ", std)

    g1 = Gaussian1D(amplitude=0.5*np.ma.max(y), mean=0.9*kwargs["halfwidth"], stddev=0.5*beam_size_pixels_stddev,
                    bounds={"amplitude": (min_amp_coeff*np.max(y), 1.),
                            "mean": (10, 2*kwargs["halfwidth"]-10),
                            "stddev": (0.25*beam_size_pixels_stddev, 5*beam_size_pixels_stddev)})
    g2 = Gaussian1D(amplitude=0.5*np.ma.max(y), mean=1.1*kwargs["halfwidth"], stddev=0.5*beam_size_pixels_stddev,
                    bounds={"amplitude": (min_amp_coeff*np.max(y), 1.),
                            "mean": (10, 2*kwargs["halfwidth"]-10),
                            "stddev": (0.25*beam_size_pixels_stddev, 5*beam_size_pixels_stddev)})
    g = g1 + g2


    if kwargs["use"] == "dlib":
        print("Using DLIB")
        lower_bounds = [0.05*np.ma.max(y), 10., 0.25*beam_size_pixels_stddev,
                        0.05*np.ma.max(y), kwargs["halfwidth"]-1, 0.25*beam_size_pixels_stddev]
        upper_bounds = [1., kwargs["halfwidth"]+1, 3*beam_size_pixels_stddev,
                        1., 2*kwargs["halfwidth"]-10, 3*beam_size_pixels_stddev]

        n_fun_eval = 3000
        def minfunc(*p):
            amp0, mean0, sigma0, amp1, mean1, sigma1 = p
            g[0].amplitude = amp0
            g[0].mean = mean0
            g[0].stddev = sigma0
            g[1].amplitude = amp1
            g[1].mean = mean1
            g[1].stddev = sigma1
            result = np.sum((y - g(x))**2)

            return result

        delta_bound, _ = dlib.find_min_global(minfunc, bound1=lower_bounds, bound2=upper_bounds, num_function_calls=n_fun_eval)
        print("delta_bound :", delta_bound)
        p = delta_bound
        print("after :",  p)
        amp0, mean0, sigma0, amp1, mean1, sigma1 = p
        g[0].amplitude = amp0
        g[0].mean = mean0
        g[0].stddev = sigma0
        g[1].amplitude = amp1
        g[1].mean = mean1
        g[1].stddev = sigma1

        x0 = [amp0, mean0, sigma0, amp1, mean1, sigma1]


    elif kwargs["use"] == "bobyqa":
        lower_bounds = [0.05*np.ma.max(y), 10., 0.25*beam_size_pixels_stddev,
                        0.05*np.ma.max(y), kwargs["halfwidth"]-1, 0.25*beam_size_pixels_stddev]
        upper_bounds = [1., kwargs["halfwidth"]+1, 3*beam_size_pixels_stddev,
                        1., 2*kwargs["halfwidth"]-10, 3*beam_size_pixels_stddev]
        def minfunc(p):
            amp0, mean0, sigma0, amp1, mean1, sigma1 = p
            g[0].amplitude = amp0
            g[0].mean = mean0
            g[0].stddev = sigma0
            g[1].amplitude = amp1
            g[1].mean = mean1
            g[1].stddev = sigma1
            result = np.sum((y - g(x))**2)
            return result

        x0 = kwargs["x0"]
        soln = pybobyqa.solve(minfunc, x0, maxfun=1000, bounds=(lower_bounds, upper_bounds),
                              seek_global_minimum=True, print_progress=False, scaling_within_bounds=True,
                              rhoend=1e-8)
        best_params = soln.x
        print(soln.f)
        print(best_params)
        x0 = best_params

    else:
        print("Using LevMarLSQFitter")
        fit_g = fitting.LevMarLSQFitter()
        g = fit_g(g, x, y, weights=1/std, maxiter=10000, acc=1e-10, estimate_jacobian=False)
        x0 = None

    # Find maximum of the fitted function
    from scipy.optimize import fmin, minimize_scalar
    max_x = fmin(lambda x: -g(x), np.argmax(y))[0]
    max_val = g(max_x)
    print("Maximum x ", max_x)
    print("Maximum value = ", max_val)

    # Find points where g = 0.25 g_max
    # solution_1 = minimize_scalar(lambda x: np.abs(g(x)-0.25*max_val), bounds=[0,max_x], method='Bounded')
    # solution_2 = minimize_scalar(lambda x: np.abs(g(x)-0.25*max_val), bounds=[max_x, 2*kwargs["halfwidth"]], method='Bounded')


    # First find upper bound
    lower_bounds = [0]
    upper_bounds = [np.where((y - 0.1*max_val) > 0)[0][0] + 5]
    n_fun_eval = 200
    delta_bound, _ = dlib.find_min_global(lambda x: np.abs(g(x)-0.1*max_val), lower_bounds, upper_bounds, n_fun_eval)
    solution_1 = delta_bound[0]
    lower_bounds = [max_x]
    upper_bounds = [np.where((y - 0.1*max_val) > 0)[0][-1] + 5]
    delta_bound, _ = dlib.find_min_global(lambda x: np.abs(g(x)-0.1*max_val), lower_bounds, upper_bounds, n_fun_eval)
    solution_2 = delta_bound[0]

    print("FWQM : ", solution_1, solution_2)

    if do_plot:
        fig, axes = plt.subplots(1, 1)
        axes.scatter(x, y)
        axes.plot(x, g[0](x), label="1")
        axes.plot(x, g[1](x), label="2")
        axes.plot(x, g(x), label="all")
        axes.axhline(0.1*max_val, lw=1, color="k")
        axes.axhline(max_val, lw=1, ls="-.", color="k")
        axes.axhline(std, lw=1, ls="--", color="k", label=r"$\sigma$")
        axes.axvline(solution_1, lw=1, color="k")
        axes.axvline(solution_2, lw=1, color="k")
        plt.legend()
        fig.savefig(os.path.join(kwargs["save_dir"], "slice_z_obs_{:.2f}.png".format(kwargs["z_obs_mas"])), dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close()


    return solution_2 - solution_1, x0


def fit_single_slice_Gauss4(x, y, **kwargs):

    do_plot = kwargs["plot"]
    beam_size_pixels_stddev = gaussian_fwhm_to_sigma*kwargs["beam_size_pixels"]
    std = kwargs["std"]
    min_amp_coeff = kwargs["min_amp_coeff"]
    print("std = ", std)

    g1 = Gaussian1D(amplitude=0.8*np.ma.max(y), mean=0.7*kwargs["halfwidth"], stddev=1.1*beam_size_pixels_stddev,
                    bounds={"amplitude": (min_amp_coeff*np.ma.max(y), 1.),
                            "mean": (10, 60),
                            "stddev": (beam_size_pixels_stddev, 5*beam_size_pixels_stddev)})
    g2 = Gaussian1D(amplitude=0.8*np.ma.max(y), mean=0.9*kwargs["halfwidth"], stddev=1.1*beam_size_pixels_stddev,
                    bounds={"amplitude": (kwargs["std"], 1.),
                            "mean": (10, 2*kwargs["halfwidth"]-10),
                            "stddev": (beam_size_pixels_stddev, 5*beam_size_pixels_stddev)})
    g3 = Gaussian1D(amplitude=0.5*np.ma.max(y), mean=1.0*kwargs["halfwidth"], stddev=1.1*beam_size_pixels_stddev,
                    bounds={"amplitude": (min_amp_coeff*np.max(y), 1.),
                            "mean": (10, 2*kwargs["halfwidth"]-10),
                            "stddev": (beam_size_pixels_stddev, 5*beam_size_pixels_stddev)})
    g4 = Gaussian1D(amplitude=1.0*np.ma.max(y), mean=1.3*kwargs["halfwidth"], stddev=1.1*beam_size_pixels_stddev,
                    bounds={"amplitude": (min_amp_coeff*np.ma.max(y), 1.),
                            "mean": (40, 90),
                            "stddev": (beam_size_pixels_stddev, 5*beam_size_pixels_stddev)})
    g = g1 + g2 + g3 + g4


    if kwargs["use"] == "dlib":
        print("Using DLIB")
        lower_bounds = [100*kwargs["std"], 40., beam_size_pixels_stddev,
                        kwargs["std"], 40., beam_size_pixels_stddev,
                        kwargs["std"], 40., beam_size_pixels_stddev,
                        kwargs["std"], 40., beam_size_pixels_stddev]
        upper_bounds = [10., kwargs["halfwidth"]+10, 2*beam_size_pixels_stddev,
                        10., kwargs["halfwidth"]+10, 2*beam_size_pixels_stddev,
                        10., kwargs["halfwidth"]+10, 2*beam_size_pixels_stddev,
                        10., kwargs["halfwidth"]+10, 2*beam_size_pixels_stddev]

        n_fun_eval = 3000
        def minfunc(*p):
            amp0, mean0, sigma0, amp1, mean1, sigma1, amp2, mean2, sigma2, amp3, mean3, sigma3 = p 
            g[0].amplitude = amp0
            g[0].mean = mean0
            g[0].stddev = sigma0
            g[1].amplitude = amp1
            g[1].mean = mean1
            g[1].stddev = sigma1
            g[2].amplitude = amp2
            g[2].mean = mean2
            g[2].stddev = sigma2
            g[3].amplitude = amp3
            g[3].mean = mean3
            g[3].stddev = sigma3
            result = np.sum((y - g(x))**2)

            return result

        delta_bound, _ = dlib.find_min_global(minfunc, bound1=lower_bounds, bound2=upper_bounds, num_function_calls=n_fun_eval)
        print("delta_bound :", delta_bound)
        p = delta_bound
        print("after :",  p)
        amp0, mean0, sigma0, amp1, mean1, sigma1, amp2, mean2, sigma2, amp3, mean3, sigma3 = p 
        g[0].amplitude = amp0
        g[0].mean = mean0
        g[0].stddev = sigma0
        g[1].amplitude = amp1
        g[1].mean = mean1
        g[1].stddev = sigma1
        g[2].amplitude = amp2
        g[2].mean = mean2
        g[2].stddev = sigma2
        g[3].amplitude = amp3
        g[3].mean = mean3
        g[3].stddev = sigma3

        x0 = p

    elif kwargs["use"] == "bobyqa":
        # lower_bounds = [0.05*np.ma.max(y), 10., 0.25*beam_size_pixels_stddev,
        #                 0.05*np.ma.max(y), kwargs["halfwidth"]-1, 0.25*beam_size_pixels_stddev]
        # upper_bounds = [1., kwargs["halfwidth"]+1, 3*beam_size_pixels_stddev,
        #                 1., 2*kwargs["halfwidth"]-10, 3*beam_size_pixels_stddev]

        lower_bounds = [kwargs["std"], 10., beam_size_pixels_stddev,
                        kwargs["std"], 10., beam_size_pixels_stddev,
                        kwargs["std"], 10., beam_size_pixels_stddev,
                        kwargs["std"], 10., beam_size_pixels_stddev]
        upper_bounds = [10., 2*kwargs["halfwidth"]-10, 4*beam_size_pixels_stddev,
                        10., 2*kwargs["halfwidth"]-10, 4*beam_size_pixels_stddev,
                        10., 2*kwargs["halfwidth"]-10, 4*beam_size_pixels_stddev,
                        10., 2*kwargs["halfwidth"]-10, 4*beam_size_pixels_stddev]

        def minfunc(p):
            amp0, mean0, sigma0, amp1, mean1, sigma1, amp2, mean2, sigma2, amp3, mean3, sigma3 = p
            g[0].amplitude = amp0
            g[0].mean = mean0
            g[0].stddev = sigma0
            g[1].amplitude = amp1
            g[1].mean = mean1
            g[1].stddev = sigma1
            g[2].amplitude = amp2
            g[2].mean = mean2
            g[2].stddev = sigma2
            g[3].amplitude = amp3
            g[3].mean = mean3
            g[3].stddev = sigma3
            result = np.sum((y - g(x))**2)

            return result

        x0 = kwargs["x0"]
        soln = pybobyqa.solve(minfunc, x0, maxfun=1000, bounds=(lower_bounds, upper_bounds),
                              seek_global_minimum=True, print_progress=False, scaling_within_bounds=True,
                              rhoend=1e-8)

        p = soln.x
        print("after :",  p)
        amp0, mean0, sigma0, amp1, mean1, sigma1, amp2, mean2, sigma2, amp3, mean3, sigma3 = p
        g[0].amplitude = amp0
        g[0].mean = mean0
        g[0].stddev = sigma0
        g[1].amplitude = amp1
        g[1].mean = mean1
        g[1].stddev = sigma1
        g[2].amplitude = amp2
        g[2].mean = mean2
        g[2].stddev = sigma2
        g[3].amplitude = amp3
        g[3].mean = mean3
        g[3].stddev = sigma3

        x0 = p

    else:
        print("Using LevMarLSQFitter")
        fit_g = fitting.LevMarLSQFitter()
        g = fit_g(g, x, y, weights=1/std, maxiter=10000, acc=1e-10, estimate_jacobian=False)
        x0 = None

    # Find maximum of the fitted function
    from scipy.optimize import fmin, minimize_scalar
    max_x = fmin(lambda x: -g(x), np.argmax(y))[0]
    max_val = g(max_x)
    x_4_med = x[g(x) > 0.1*max_val]
    med_val = np.median(g(x_4_med))
    print("Maximum x ", max_x)
    print("Maximum value = ", max_val)
    print("Median value = ", med_val)

    # Value at which to calculate width ========================================
    cut_value = 0.1*max_val
    # cut_value = 0.25*med_val
    # ==========================================================================

    # Find points where g = 0.25 g_max
    # solution_1 = minimize_scalar(lambda x: np.abs(g(x)-0.25*max_val), bounds=[0,max_x], method='Bounded')
    # solution_2 = minimize_scalar(lambda x: np.abs(g(x)-0.25*max_val), bounds=[max_x, 2*kwargs["halfwidth"]], method='Bounded')


    # First find upper bound
    lower_bounds = [0]
    upper_bounds = [np.where((y - cut_value) > 0)[0][0] + 5]
    n_fun_eval = 200
    delta_bound, _ = dlib.find_min_global(lambda x: np.abs(g(x)-cut_value), lower_bounds, upper_bounds, n_fun_eval)
    solution_1 = delta_bound[0]
    lower_bounds = [max_x]
    upper_bounds = [np.where((y - cut_value) > 0)[0][-1] + 5]
    delta_bound, _ = dlib.find_min_global(lambda x: np.abs(g(x)-cut_value), lower_bounds, upper_bounds, n_fun_eval)
    solution_2 = delta_bound[0]

    print("FWQM : ", solution_1, solution_2)

    if do_plot:
        fig, axes = plt.subplots(1, 1)
        axes.scatter(x, y)
        axes.plot(x, g[0](x), label="1")
        axes.plot(x, g[1](x), label="2")
        axes.plot(x, g[2](x), label="3")
        axes.plot(x, g[3](x), label="4")
        axes.plot(x, g(x), label="all")
        axes.axhline(cut_value, lw=1, color="k")
        axes.axhline(max_val, lw=1, ls="-.", color="k")
        axes.axhline(std, lw=1, ls="--", color="k", label=r"$\sigma$")
        axes.axvline(solution_1, lw=1, color="k")
        axes.axvline(solution_2, lw=1, color="k")
        plt.legend()
        fig.savefig(os.path.join(kwargs["save_dir"], "slice_z_obs_{:.2f}.png".format(kwargs["z_obs_mas"])), dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close()

    return solution_2 - solution_1, x0


def fit_profile(pos_to_plot, widths_to_plot, save_prefix="profile_fit", save_dir=None, fig=None, fix_r0=False):
    if save_dir is None:
        save_dir = os.getcwd()

    @custom_model
    def power_law(r, amp=0.5, r0=0.0, k=0.5):
        return amp*(r + r0)**k

    if fix_r0:
        pl_init = power_law(fixed={"r0": True})
    else:
        pl_init = power_law(bounds={"r0": (-0.4, 0.4)})
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

    # I didn't use this in M87 paper draft
    # axes[1].set_xscale("log", base=10.)
    # axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))

    # Some backup options
    # axes[1].xaxis.set_major_formatter(ScalarFormatter())
    # axes[1].xaxis.set_minor_formatter(ticker.FuncFormatter(lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    fig.subplots_adjust(hspace=0)

    if save_prefix is not None:
        fig.savefig(os.path.join(save_dir, "{}.png".format(save_prefix)), bbox_inches="tight", dpi=300)
    plt.show()

    return fig


def create_deep_stack(epochs, uvfits_files_dir, common_circ_beam_size_mas = 1.0, common_mapsize=(2048, 0.1), boxfile=None):
    images = list()
    for epoch in epochs:
        uvfits = os.path.join(uvfits_files_dir, "1228+126.u.{}.uvf".format(epoch))
        outname = os.path.join(uvfits_files_dir, "{}_cc.fits".format(epoch))

        CLEAN_difmap(uvfits, "I", common_mapsize, outname, restore_beam=(common_circ_beam_size_mas, common_circ_beam_size_mas, 0),
                     boxfile=boxfile, working_dir=None, uvrange=None,
                     box_clean_nw_niter=1000, clean_gain=0.03, dynam_su=20, dynam_u=6, deep_factor=1.0,
                     remove_difmap_logs=True, save_noresid=None, save_resid_only=None, save_dfm=None,
                     noise_to_use="F", shift=None)

        image = create_image_from_fits_file(outname)
        images.append(image.image)
    stack = np.mean(images, axis=0)
    return stack


def construct_stack_from_CCFITS(ccfits_files, uvfits_files, common_mapsize, common_circ_beam_size_mas,
                                working_dir=None, stokes="I", blc=None, trc=None):
    if working_dir is None:
        working_dir = os.getcwd()
    # tmp_difmap_mdl_file = os.path.join(working_dir, "dfm.mdl")
    # images = list()
    # for ccfits, uvfits in zip(ccfits_files, uvfits_files):
    #     CCFITS_to_difmap(ccfits, tmp_difmap_mdl_file)
    #     out_ccfits = os.path.join(working_dir, "cc.fits")
    #     convert_difmap_model_file_to_CCFITS(tmp_difmap_mdl_file, stokes, common_mapsize,
    #                                         (common_circ_beam_size_mas, common_circ_beam_size_mas, 0),
    #                                         uvfits, out_ccfits, shift=None, show_difmap_output=True)
    #     image = create_image_from_fits_file(out_ccfits)
    #     images.append(image.image)
    # stack = np.mean(images, axis=0)

    stack_file = "deep_stack.txt"
    save_dir = "/home/ilya/github/stack_fitter/simulations"
    stack = np.loadtxt(os.path.join(save_dir, stack_file))

    npixels_beam = np.pi*common_circ_beam_size_mas/(4*np.log(2)*common_mapsize[1]**2)
    std = 0.06e-03
    if blc is None or trc is None:
        blc, trc = find_bbox(stack, level=4*std, min_maxintensity_mjyperbeam=300*std,
                             min_area_pix=100*npixels_beam, delta=10)
    print(blc, trc)

    # IPOL contours
    ccimage = create_image_from_fits_file(os.path.join(working_dir, "cc.fits"))
    fig = iplot(stack, x=ccimage.x, y=ccimage.y,
                min_abs_level=1*std, blc=blc, trc=trc, beam=(common_circ_beam_size_mas, common_circ_beam_size_mas, 0),
                close=True, show_beam=True, show=False,
                contour_color='gray', contour_linewidth=0.25)
    axes = fig.get_axes()[0]
    fig.savefig(os.path.join(save_dir, "observed_stack_i_deep.png"), dpi=600, bbox_inches="tight")

    return stack


if __name__ == "__main__":


    save_dir = "/home/ilya/github/stack_fitter/real"
    # save_dir = "/home/ilya/github/stack_fitter/simulations"

    # # uvfits_files_dir = "/home/ilya/Downloads/M87_uvf"
    # # epochs = ("2000_12_30",
    # #           "2000_05_08",
    # #           "2000_01_22",
    # #           "2009_05_23")
    # # uvfits_files = [os.path.join(uvfits_files_dir, "1228+126.u.{}.uvf".format(epoch)) for epoch in epochs]
    # # ccfits_files = [os.path.join(save_dir, "1228+126.u.{}.icn.fits".format(epoch)) for epoch in epochs]
    # # common_mapsize = (2048, 0.1)
    # # common_circ_beam_size_mas = 1.0
    # # stack_image = construct_stack_from_CCFITS(ccfits_files, uvfits_files, common_mapsize,
    # #                                           common_circ_beam_size_mas, working_dir=save_dir, stokes="I")
    # #
    # # np.savetxt(os.path.join(save_dir, "deep_stack.txt"), stack_image)
    # #
    # # sys.exit(0)
    #

    # n_epochs = 45
    # # stack_file = f"stack_i_{n_epochs}_epochs.txt"
    # stack_file = f"stack_i_{n_epochs}_epochs_inclined.txt"
    # image = np.loadtxt(os.path.join(save_dir, stack_file))

    image = pf.getdata("/home/ilya/github/stack_fitter/real/1228+126.u.stacked.i.fits.gz").squeeze()
    zs, Rs, positions, slices = get_slices(image, pixsize_mas=0.1, beam_size_mas=0.85, save_dir=save_dir, dlib_max=0.6,
                                           z_obs_min_mas=0.5, z_obs_max_mas=20., rotation_angle_deg=17.0, plot=True)
    #
    np.savetxt(os.path.join(save_dir, "positions.dat"), np.array(positions))
    np.savetxt(os.path.join(save_dir, "slices.dat"), np.atleast_2d(slices))
    slices_original = slices

    save_file = "za_rs_median.txt"
    np.savetxt(os.path.join(save_dir, save_file), np.vstack((zs, Rs)).T)
    zs, Rs = np.loadtxt(os.path.join(save_dir, save_file), unpack=True)
    fit_profile(zs, Rs, save_dir=save_dir, fix_r0=False, save_prefix="profile_fit")

    sys.exit(0)




    slices_original = np.loadtxt(os.path.join(save_dir, "slices.dat"))
    slices_all = list()
    for angle in np.linspace(17-2, 17+2, 20):
        print("Anlge = ", angle)
        try:
            zs, Rs, positions, slices = get_slices(image, pixsize_mas=0.1, beam_size_mas=0.86, save_dir=save_dir,
                                                   dlib_max=0.6, z_obs_min_mas=0.5, z_obs_max_mas=20., rotation_angle_deg=angle, plot=False)
        except:
            continue
        slices_all.append(slices)


    slices_std = list()
    for i in range(len(slices_original)):
        slices_std.append(np.std([s[i] for s in slices_all], axis=0))

    np.savetxt(os.path.join(save_dir, "slices_std.dat"), np.atleast_2d(slices_std))

    # Rs_all = list()
    # for angle in np.linspace(17-2, 17+2, 20):
    #     zs, Rs = get_slices(image, pixsize_mas=0.1, beam_size_mas=0.86, save_dir=save_dir,
    #                         z_obs_min_mas=0.5, z_obs_max_mas=20., rotation_angle_deg=angle, plot=False)
    #     Rs_all.append(Rs)
    #
    # Rs_err = mad_std(Rs_all, axis=0)
    #
    # zs, Rs = get_slices(image, pixsize_mas=0.1, beam_size_mas=0.86, save_dir=save_dir,
    #                     z_obs_min_mas=0.5, z_obs_max_mas=20., rotation_angle_deg=17, plot=False)
    #
    # fig, axes = plt.subplots(1, 1)
    # axes.errorbar(zs, Rs, yerr=Rs_err, fmt=".", color="k")
    # plt.show()


    # np.savetxt(os.path.join(save_dir, "zs_rs.txt"), np.vstack((zs, Rs)).T)
    # zs, Rs = np.loadtxt(os.path.join(save_dir, "zs_rs.txt"), unpack=True)
    # fit_profile(zs, Rs, save_dir=save_dir, fix_r0=False, save_prefix="profile_fit")
