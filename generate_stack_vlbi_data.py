import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import glob
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import sys
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from spydiff import clean_difmap, find_image_std, find_bbox, find_nw_beam
from uv_data import UVData, downscale_uvdata_by_freq
from from_fits import create_clean_image_from_fits_file
from image import plot as iplot
sys.path.insert(0, "../bk_transfer")
from jet_image import JetImage, TwinJetImage


freq_ghz = 15.4
# Directory to save files
save_dir = "/home/ilya/github/stack_fitter/simulations"
# Some template UVFITS with full polarization. Its uv-coverage and noise will be used while creating fake data
# template_uvfits = "/home/ilya/github/bk_transfer/uvfits/1458+718.u.2006_09_06.uvf"
template_uvfits_files = glob.glob(os.path.join("/home/ilya/Downloads/M87_uvf/", "*uvf"))
# Multiplicative factor for noise added to model visibilities.
noise_scale_factor = 1.0
# Used in CLEAN
mapsize = (1024, 0.1)
# Common beam
common_beam = (1.35, 1.35, 0)
model_jet_image = "/home/ilya/github/stack_fitter/simulations/jet_image_i_15.4_2ridges.txt"
model_cjet_image = "/home/ilya/github/stack_fitter/simulations/cjet_image_i_15.4_2ridges.txt"
# Make jet along RA
rot_angle_deg = -90.
# Real M87
# rot_angle_deg = -107.

# C++ code run parameters
z = 0.00436
n_along = 1400
n_across = 500
lg_pixel_size_mas_min = -1.5
lg_pixel_size_mas_max = -1.5
resolutions = np.logspace(lg_pixel_size_mas_min, lg_pixel_size_mas_max, n_along)
print("Model jet extends up to {:.1f} mas!".format(np.sum(resolutions)))
##############################################
# No need to change anything below this line #
##############################################

# Plot only jet emission and do not plot counter-jet?
jet_only = False
path_to_script = "/home/ilya/github/bk_transfer/scripts/script_clean_rms"

images_i = list()
images_q = list()
images_u = list()
images_pang = list()
stokes = "I"

for i in range(5):
    # -107 for M87; -90 for making it along RA axis.
    uvfits = template_uvfits_files[i]
    print("File : ", uvfits)
    uvdata = UVData(uvfits)
    # if "RR" not in uvdata.stokes:
    #     stokes = "LL"
    # else:
    #     stokes = "I"
    noise = uvdata.noise(average_freq=False, use_V=False)
    # If one needs to decrease the noise this is the way to do it
    for baseline, baseline_noise_std in noise.items():
        noise.update({baseline: noise_scale_factor*baseline_noise_std})

    jm = JetImage(z=z, n_along=n_along, n_across=n_across,
                    lg_pixel_size_mas_min=lg_pixel_size_mas_min, lg_pixel_size_mas_max=lg_pixel_size_mas_max,
                    jet_side=True, rot=np.deg2rad(rot_angle_deg))
    cjm = JetImage(z=z, n_along=n_along, n_across=n_across,
                     lg_pixel_size_mas_min=lg_pixel_size_mas_min, lg_pixel_size_mas_max=lg_pixel_size_mas_max,
                     jet_side=False, rot=np.deg2rad(rot_angle_deg))
    jm.load_image_stokes(stokes, model_jet_image, scale=1.0)
    cjm.load_image_stokes(stokes, model_cjet_image, scale=1.0)

    # List of models (for J & CJ) for all stokes
    js = TwinJetImage(jm, cjm)

    uvdata.zero_data()
    rotated_uvdata = uvdata.create_uvfits_with_rotated_uv(np.deg2rad(17), os.path.join(save_dir, "template.uvf"), overwrite=True)
    if jet_only:
        rotated_uvdata.substitute([jm])
    else:
        rotated_uvdata.substitute([js])
        # pass
    # Optionally
    # uvdata.rotate_evpa(np.deg2rad(rot_angle_deg))
    rotated_uvdata.noise_add(noise)
    # uvdata.create_uvfits_with_rotated_uv(np.deg2rad(17), os.path.join(save_dir, "template.uvf"), overwrite=True)
    need_downscale_uv = downscale_uvdata_by_freq(rotated_uvdata)
    print("Need downscale = ", need_downscale_uv)
    rotated_uvdata.save(os.path.join(save_dir, "template.uvf"), rewrite=True, downscale_by_freq=need_downscale_uv)
    nw_beam = find_nw_beam(os.path.join(save_dir, "template.uvf"), mapsize=(1024, 0.1), stokes=stokes)
    print("NW beam = ", nw_beam)
    # uvdata.save(os.path.join(save_dir, "template.uvf"), rewrite=True)

    # CLEAN synthetic UV-data
    outfname = "model_cc_i.fits"
    if os.path.exists(os.path.join(save_dir, outfname)):
        os.unlink(os.path.join(save_dir, outfname))
    clean_difmap(fname="template.uvf", path=save_dir,
                 outfname=outfname, outpath=save_dir, stokes=stokes.lower(),
                 mapsize_clean=mapsize, path_to_script=path_to_script,
                 show_difmap_output=True,
                 # text_box=text_boxes[freq],
                 dfm_model=os.path.join(save_dir, "model_cc_i.mdl"),
                 beam_restore=common_beam)

    ccimage = create_clean_image_from_fits_file(os.path.join(save_dir, "model_cc_i.fits"))
    ipol = ccimage.image
    # (bmaj [mas], bmin [mas], bpa [rad])
    beam = ccimage.beam
    # Number of pixels in beam
    npixels_beam = np.pi*beam[0]*beam[1]/(4*np.log(2)*mapsize[1]**2)

    std = find_image_std(ipol, beam_npixels=npixels_beam)
    print("IPOL image std = {} mJy/beam".format(1000*std))
    blc, trc = find_bbox(ipol, level=4*std, min_maxintensity_mjyperbeam=10*std,
                         min_area_pix=4*npixels_beam, delta=10)
    if blc[0] == 0: blc = (blc[0]+1, blc[1])
    if blc[1] == 0: blc = (blc[0], blc[1]+1)
    if trc[0] == ipol.shape: trc = (trc[0]-1, trc[1])
    if trc[1] == ipol.shape: trc = (trc[0], trc[1]-1)

    images_i.append(ipol)


stack_i = np.mean(images_i, axis=0)
np.savetxt(os.path.join(save_dir, "stack_i.txt"), stack_i)
std = find_image_std(stack_i, beam_npixels=npixels_beam)
blc, trc = find_bbox(stack_i, level=4*std, min_maxintensity_mjyperbeam=30*std,
                     min_area_pix=20*npixels_beam, delta=10)

# IPOL contours
fig = iplot(stack_i, x=ccimage.x, y=ccimage.y,
            min_abs_level=4*std, blc=blc, trc=trc, beam=(beam[0], beam[1], np.rad2deg(beam[2])), close=True, show_beam=True, show=False,
            contour_color='gray', contour_linewidth=0.25)
axes = fig.get_axes()[0]
fig.savefig(os.path.join(save_dir, "observed_stack_i.png"), dpi=600, bbox_inches="tight")
