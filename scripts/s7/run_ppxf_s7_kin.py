import matplotlib
matplotlib.use("agg")

import sys, os 
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
from numpy.random import RandomState
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm
from time import time

from ppxftests.run_ppxf import run_ppxf

from settings import CLEAN, ppxf_output_path, fig_path, Aperture, get_aperture_coords, gals_all
from load_galaxy_spectrum import load_galaxy_spectrum

savefigs = True

# Keyword args for ppxf - should be consistent between runs
kwargs = {
        "isochrones": "Padova", 
        "z": 0.0, 
        "fit_gas": False, 
        "ngascomponents": 0,
        "fit_agn_cont": False,
        "reddening": None, 
        "mdegree": -1, 
        "adegree": 12,
        "regularisation_method": "none",
        "interactive_mode": False,
}

###########################################################################
# Helper function for running MC simulations
###########################################################################
def ppxf_mc_helper(args):
    """
    Helper function used in ppxf_mc() for running Monte Carlo simulations 
    with ppxf. Note that no regularisation is used.

    Inputs:
    args        list containing the following:
        seed                integer required to seed the RNG for computing the extra 
                            noise to be added to the spectrum
        spec                input (noisy) spectrum
        spec_err            corresponding 1-sigma errors 
        lambda_vals_A       wavelength values (Angstroms)
        bad_pixel_ranges_A  Bad wavelength ranges to mask out during the fit

    """

    # Unpack arguments
    seed, spec, spec_err, lambda_vals_A, bad_pixel_ranges_A = args
    
    # Add "extra" noise to the spectrum
    rng = RandomState(seed)
    noise = rng.normal(scale=spec_err)
    spec_noise = spec + noise

    # This is to mitigate the "edge effects" of the convolution with the LSF
    spec_noise[0] = -9999
    spec_noise[-1] = -9999

    # If any noise values are NaN, these make the corresponding spec values NaN too.
    # Because ppxf() throws an error if there are any NaNs in the input (even if these values get masked out by good_px) we need to 
    # remove these. Therefore, we set these values to -9999 as above - in run_ppxf(), -ve values 
    # in spec_log get masked out so this should be OK.
    spec_noise[~np.isfinite(spec_noise)] = -9999
    
    # Run ppxf
    pp = run_ppxf(spec=spec_noise, spec_err=spec_err, lambda_vals_A=lambda_vals_A,
                  bad_pixel_ranges_A=bad_pixel_ranges_A, clean=CLEAN,
                  **kwargs)
    return pp

###########################################################################
def ppxf_mc(ppxf_fn, spec, spec_err, lambda_vals_A, bad_pixel_ranges_A, niters, nthreads):
    """
    Run Monte-Carlo simulations with ppxf.

    Run ppxf a total of niters times on the input spectrum (spec), each time 
    adding additional random noise governed by the 1-sigma errors on the 
    input spectrum (spec_err)

    Inputs:
    ppxf_fn             Function to call
    spec                Input spectrum
    spec_err            Corresponding 1-sigma errors
    lambda_vals_A       Wavelength values for the spectrum (Angstroms)
    bad_pixel_ranges_A  Bad wavelength ranges to mask out during the fit
    niters              Total number of MC iterations 
    nthreads            Number of threads used
    
    """
    # Input arguments
    seeds = list(np.random.randint(low=0, high=100 * niters, size=niters))
    args_list = [[s, spec, spec_err, lambda_vals_A, bad_pixel_ranges_A] for s in seeds]

    # Run in parallel
    if nthreads > 1:
        print(f"ppxf_mc(): running ppxf on {nthreads} threads...")
        t = time()
        with multiprocessing.Pool(nthreads) as pool:
            pp_list = list(tqdm(pool.imap(ppxf_fn, args_list), total=niters))
        print(f"ppxf_mc(): elapsed time = {time() - t:.2f} s")
    else:
        print(f"ppxf_mc(): running ppxf sequentially...")
        pp_list = []
        for args in tqdm(args_list):
            pp_list.append(ppxf_fn(args))

    return pp_list

###########################################################################
def ppxf_mc_helper_VdS(args):
    """
    Helper function used in ppxf_mc() for running Monte Carlo simulations 
    with ppxf. Note that no regularisation is used.
    This function uses the method of van de Sande et al. (2017) to 
    estimate errors, rather than my previously used method.

    Inputs:
    args        list containing the following:
        seed                integer required to seed the RNG for computing the extra 
                            noise to be added to the spectrum
        spec                input (noisy) spectrum
        spec_err            corresponding 1-sigma errors 
        lambda_vals_A       wavelength values (Angstroms)
        bad_pixel_ranges_A  Bad wavelength ranges to mask out during the fit

    """

    # Unpack arguments
    seed, fit, spec, spec_err, lambda_vals_A, bad_pixel_ranges_A = args

    # Random state generator
    rng = RandomState(seed)
    
    # Shuffle residuals & add back to the fit  
    residuals = fit - spec
    n_chunk = 8
    n = len(lambda_vals_A) // n_chunk
    noise = np.copy(residuals)
    for ii in range(n_chunk):
        rng.shuffle(noise[ii * n:(ii + 1) * n])
    rng.shuffle(noise[(ii + 1) * n:])
    spec_noise = fit + noise

    # This is to mitigate the "edge effects" of the convolution with the LSF
    spec_noise[0] = -9999
    spec_noise[-1] = -9999

    # If any noise values are NaN, these make the corresponding spec values NaN too.
    # Because ppxf() throws an error if there are any NaNs in the input (even if these values get masked out by good_px) we need to 
    # remove these. Therefore, we set these values to -9999 as above - in run_ppxf(), -ve values 
    # in spec_log get masked out so this should be OK.
    spec_noise[~np.isfinite(spec_noise)] = -9999
    
    # Run ppxf on the best fit + the shuffled residuals
    pp = run_ppxf(spec=spec_noise, spec_err=spec_err, lambda_vals_A=lambda_vals_A,
                  bad_pixel_ranges_A=bad_pixel_ranges_A, clean=CLEAN,
                  **kwargs)
    return pp

###########################################################################
def ppxf_mc_VdS(ppxf_fn, spec, spec_err, lambda_vals_A, bad_pixel_ranges_A, niters, nthreads):
    """
    Run Monte-Carlo simulations with ppxf.

    Run ppxf a total of niters times on the input spectrum (spec), each time 
    adding additional random noise governed by the 1-sigma errors on the 
    input spectrum (spec_err)

    This function uses the method of van de Sande et al. (2017) to 
    estimate errors, rather than my previously used method.

    Inputs:
    ppxf_fn             Function to call
    spec                Input spectrum
    spec_err            Corresponding 1-sigma errors
    lambda_vals_A       Wavelength values for the spectrum (Angstroms)
    bad_pixel_ranges_A  Bad wavelength ranges to mask out during the fit
    niters              Total number of MC iterations 
    nthreads            Number of threads used
    
    """
    # Run ppxf once to get a fit 
    pp = run_ppxf(spec=spec, spec_err=spec_err, lambda_vals_A=lambda_vals_A,
                  bad_pixel_ranges_A=bad_pixel_ranges_A, clean=CLEAN,
                  **kwargs)

    # Interpolate fit back to linear wavelength grid so we can subtract it from the input to get the residuals
    fit = np.full_like(lambda_vals_A, -9999)
    fit[:len(lambda_vals_A) - 1] = interp1d(x=pp.lam, y=pp.bestfit * pp.norm)(lambda_vals_A[:-1])

    # Input arguments
    seeds = list(np.random.randint(low=0, high=100 * niters, size=niters))
    args_list = [[seed, fit, spec, spec_err, lambda_vals_A, bad_pixel_ranges_A] for seed in seeds]

    # Run in parallel
    if nthreads > 1:
        print(f"ppxf_mc(): running ppxf on {nthreads} threads...")
        t = time()
        with multiprocessing.Pool(nthreads) as pool:
            pp_list = list(tqdm(pool.imap(ppxf_fn, args_list), total=niters))
        print(f"ppxf_mc(): elapsed time = {time() - t:.2f} s")
    else:
        print(f"ppxf_mc(): running ppxf sequentially...")
        pp_list = []
        for args in tqdm(args_list):
            pp_list.append(ppxf_fn(args))

    return pp_list

if __name__ == "__main__":

    ###########################################################################
    # User options
    ###########################################################################
    args = sys.argv
    if args[1].upper() == "DEBUG":
        debug = True  # If true, run on only 20 threads & save to a new DataFrame with 'DEBUG' in the file name
        args = args[1:]  
    else:
        debug = False

    # MC settings
    if debug:
        niters = 100
        nthreads = 10
    else:
        niters = 150  #TODO change this back to 1000... or keep it at 150 to be consistent with VdS?
        nthreads = 56

    # x and y coordinates in DataCube corresponding to the chosen aperture
    aperture = Aperture[args[1]]
    rr, cc = get_aperture_coords(aperture)

    # List of Galaxies
    if len(args) > 2:
        gals = args[2:]
    else:
        gals = gals_all

    ##############################################################################
    # Run ppxf for each galaxy & append to DataFrame
    ##############################################################################
    for gal in gals:
        df_fname = f"s7_ppxf_DEBUG_kinematics_{gal}_{aperture.name}.hd5" if debug else f"s7_ppxf_kinematics_{gal}_{aperture.name}.hd5"
        df = pd.DataFrame(dtype="object")

        print("###################################################################")
        if not debug:
            print(f"Now running on {gal} (aperture {aperture})...")
        else:
            print(f"Now running on {gal} (aperture {aperture}) in DEBUG mode...")
        print("###################################################################")
        
        ##############################################################################
        # Load the aperture spectrum
        lambda_vals_obs_A, lambda_vals_rest_A, spec, spec_cont_only, spec_err, norm, bad_pixel_ranges_A =\
            load_galaxy_spectrum(gal, aperture)

        ##############################################################################
        # PPXF: kinematic fit
        ##############################################################################
        t = time()
        print(f"Gal {gal}: running ppxf...")
        #TODO use the CLEAN keyword
        """
        Noise scaling:
        - run once with flat noise values (= average of noise across full spectrum)
        - calculate std(fit - data) := A
        - compate with mean(data_err) := B
        - scale data_err by A / B 
        - e.g. if A = 0.1, but B = 0.01, then need to boost data_err by 10 = A / B.
        """
        # Run ppxf a first time using a flat noise array. 
        pp1 = run_ppxf(spec=spec_cont_only * norm, 
                    spec_err=np.ones(spec_err.shape) * np.nanmean(spec_err * norm), 
                    lambda_vals_A=lambda_vals_rest_A,
                    bad_pixel_ranges_A=bad_pixel_ranges_A,
                    plotit=True if debug else False,
                    clean=False,
                    **kwargs)
        if debug:
            plt.gcf().get_axes()[0].set_title(f"Initial fit for {gal}: v_* = {pp1.sol[0]:.2f} km/s; sigma_* = {pp1.sol[1]:.2f} km/s")

        # Sanity check to make sure we understand the different normalisation factors to keep track of
        # Remember: pp_kin.norm is NOT the same as norm - pp_kin.norm is the normalisation factor for the log-rebinned spectrum, norm is that for the linearly-binned spectrum
        plt.figure()
        plt.plot(pp1.lam, pp1.spec_log_norm * pp1.norm)
        plt.plot(pp1.lam, pp1.bestfit * pp1.norm)
        plt.plot(lambda_vals_rest_A, spec_cont_only * norm)

        # Re-scale the noise as per van de Sande et al. (2017b)
        resid = (pp1.bestfit - pp1.spec_log_norm) * pp1.norm
        resid_std = np.nanstd(resid)
        noise_mean = np.nanmean(spec_err * norm)
        spec_err_scaled = spec_err * resid_std / noise_mean

        # Re-run ppxf with scaled noise
        pp_kin = run_ppxf(spec=spec_cont_only * norm, 
                        spec_err=spec_err_scaled * norm, 
                        lambda_vals_A=lambda_vals_rest_A,
                        bad_pixel_ranges_A=bad_pixel_ranges_A,
                        plotit=True if debug else False,
                        clean=CLEAN,
                        **kwargs)
        if debug:
            plt.gcf().get_axes()[0].set_title(f"Second fit for {gal}: v_* = {pp_kin.sol[0]:.2f} km/s; sigma_* = {pp_kin.sol[1]:.2f} km/s (noise scaled by {resid_std / noise_mean:.4f})")
        
        print(f"Gal {gal}: total time in run_ppxf: {time() - t:.2f} seconds")

        if debug:
            plt.close("all")

        #/////////////////////////////////////////////////////////////////////////////
        # Method 1: original method (add randomised noise to data each iteration)
        pp_list_old = ppxf_mc(ppxf_mc_helper, spec_cont_only * norm, spec_err_scaled * norm, 
                              lambda_vals_rest_A, bad_pixel_ranges_A, niters, nthreads)

        v_vals = [pp.sol[0] for pp in pp_list_old]
        sigma_vals = [pp.sol[1] for pp in pp_list_old]
        v_mean_old = np.nanmean(v_vals)
        v_std_old = np.nanstd(v_vals)
        sigma_mean_old = np.nanmean(sigma_vals)
        sigma_std_old = np.nanstd(sigma_vals)
        
        fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
        axs[0].hist(v_vals, bins=20, label="Standard MC")
        axs[0].axvline(np.nanmean(v_vals), color="k")
        axs[0].set_title(r"$v = %.2f \pm %.2f$ km/s" % (np.nanmean(v_vals), np.nanstd(v_vals)))
        axs[1].hist(sigma_vals, bins=20, label="Standard MC")
        axs[1].axvline(np.nanmean(sigma_vals), color="k")
        axs[1].set_title(r"$\sigma = %.2f \pm %.2f$ km/s" % (np.nanmean(sigma_vals), np.nanstd(sigma_vals)))

        #/////////////////////////////////////////////////////////////////////////////
        # Method 2: VdS method (re-run fit on best fit from initial run + shuffled residuals)
        pp_list_VdS = ppxf_mc_VdS(ppxf_mc_helper_VdS, spec_cont_only * norm, spec_err_scaled * norm, 
                                  lambda_vals_rest_A, bad_pixel_ranges_A, niters, nthreads)

        v_vals = [pp.sol[0] for pp in pp_list_VdS]
        sigma_vals = [pp.sol[1] for pp in pp_list_VdS]
        v_mean_VdS = np.nanmean(v_vals)
        v_std_VdS = np.nanstd(v_vals)
        sigma_mean_VdS = np.nanmean(sigma_vals)
        sigma_std_VdS = np.nanstd(sigma_vals)

        axs[0].hist(v_vals, bins=20, histtype="step", label="VdS")
        axs[0].axvline(np.nanmean(v_vals), color="r")
        axs[0].set_title(r"$v = %.2f \pm %.2f$ km/s" % (np.nanmean(v_vals), np.nanstd(v_vals)))
        axs[1].hist(sigma_vals, bins=20, histtype="step", label="VdS")
        axs[1].axvline(np.nanmean(sigma_vals), color="r")
        axs[1].set_title(r"$\sigma = %.2f \pm %.2f$ km/s" % (np.nanmean(sigma_vals), np.nanstd(sigma_vals)))
        
        # Save to file
        if savefigs:
            fig.savefig(os.path.join(fig_path, f"{gal}_kin_{aperture.name}.pdf"), bbox_inches="tight")
        plt.close(fig)

        ##############################################################################
        # Extract information for DataFrame
        ##############################################################################
        thisrow = {}
        # Fit parameters
        thisrow["Galaxy"] = gal
        thisrow["Emission lines included in fit?"] = pp_list_old[0].fit_gas
        thisrow["AGN continuum included in fit?"] = pp_list_old[0].fit_agn_cont
        thisrow["Extinction curve included in fit?"] = True if pp_list_old[0].reddening is not None else False
        thisrow["Multiplicative polynomial included in fit?"] = True if pp_list_old[0].mdegree > 0 else False
        thisrow["Degree of multiplicative polynomial"] = pp_list_old[0].mdegree
        thisrow["Additive polynomial included in fit?"] = True if pp_list_old[0].degree > 0 else False
        thisrow["Degree of additive polynomial"] = pp_list_old[0].degree
        thisrow["Isochrones"] = pp_list_old[0].isochrones
        thisrow["v_* (old)"] = v_mean_old
        thisrow["sigma_* (old)"] = sigma_mean_old
        thisrow["v_* error (old)"] = v_std_old
        thisrow["sigma_* error (old)"] = sigma_std_old
        thisrow["v_* (VdS)"] = v_mean_VdS
        thisrow["sigma_* (VdS)"] = sigma_mean_VdS
        thisrow["v_* error (VdS)"] = v_std_VdS
        thisrow["sigma_* error (VdS)"] = sigma_std_VdS
        df = df.append(thisrow, ignore_index=True)

        # Save to file
        df.to_hdf(os.path.join(ppxf_output_path, df_fname), key=aperture.name)