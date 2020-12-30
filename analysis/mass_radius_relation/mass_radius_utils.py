from pathlib import Path
import sys

import numpy as np
from astropy import table

# need to add the correct path to import utils
code_home_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(code_home_dir / "pipeline"))
import utils

# ======================================================================================
#
# functions for writing to an output file with fit info
#
# ======================================================================================
def write_fit_results(fit_out_file, name, number, best_fit_params, fit_history):
    """
    Write the results of one fit to a file

    :param fit_out_file: Opened file object to write these results to
    :param name: Name of the fitted sample
    :param number: Number of clusters in this sample
    :param best_fit_params: The 3 best fit parameters: slope, intercept, scatter
    :param fit_history: The history of these 3 parameters, used to find errors
    :return: None, but the info is written to the file
    """
    print_str = f"\t\t{name} & {number}"
    # the second parameter is the log of clusters at 10^4. Put it back to linear space
    best_fit_params[1] = 10 ** best_fit_params[1]
    fit_history[1] = [10 ** f for f in fit_history[1]]
    for idx in range(len(best_fit_params)):
        std = np.std(fit_history[idx])
        print_str += f" & {best_fit_params[idx]:.3f} $\pm$ {std:.3f}"
    print_str += "\\\\ \n"
    fit_out_file.write(print_str)


# ======================================================================================
#
# Data handling
#
# ======================================================================================
def make_big_table(tables_loc_list):
    """
    Read all the catalogs passed in, stack them together, and throw out bad clusters

    :param tables_loc_list: List of strings holding the paths to all the catalogs
    :return: One astropy table with all the good clusters from this sample
    """
    catalogs = []
    for item in tables_loc_list:
        this_cat = table.Table.read(item, format="ascii.ecsv")
        gal_dir = Path(item).parent.parent
        this_cat["distance"] = utils.distance(gal_dir).to("Mpc").value
        catalogs.append(this_cat)

    # then stack them together in one master catalog
    big_catalog = table.vstack(catalogs, join_type="inner")

    # filter out the clusters that can't be used in fitting the mass-radius relation
    # I did investigate some weird errors that are present. There are clusters with a
    # a fitted mass but zeros for the min and max. Throw those out. Also throw out ones
    # where the error range is improper (i.e. the min is higher than the best fit)
    mask = np.logical_and.reduce(
        [
            big_catalog["good"],
            big_catalog["mass_msun"] > 0,
            big_catalog["mass_msun_max"] > 0,
            big_catalog["mass_msun_min"] > 0,
            big_catalog["mass_msun_min"] <= big_catalog["mass_msun"],
            big_catalog["mass_msun_max"] >= big_catalog["mass_msun"],
        ]
    )
    return big_catalog[mask]


# get some commonly used items from the table and transform them to log properly
def get_my_masses(catalog, mask):
    """
    Get the masses from my catalog, along with their errors

    :param catalog: Catalog to retrieve the masses from
    :param mask: Mask to apply to the data, to restrict to certain clusters
    :return: Tuple with three elements: mass, lower mass error, upper mass error
    """
    mass = catalog["mass_msun"][mask]
    # mass errors are reported as min and max values
    mass_err_lo = mass - catalog["mass_msun_min"][mask]
    mass_err_hi = catalog["mass_msun_max"][mask] - mass

    return mass, mass_err_lo, mass_err_hi


def get_my_radii(catalog, mask):
    """
    Get the radii from my catalog, along with their errors

    :param catalog: Catalog to retrieve the radii from
    :param mask: Mask to apply to the data, to restrict to certain clusters
    :return: Tuple with three elements: radius, lower radius error, upper radius error
    """
    r_eff = catalog["r_eff_pc_rmax_15pix_best"][mask]
    r_eff_err_lo = catalog["r_eff_pc_rmax_15pix_e-"][mask]
    r_eff_err_hi = catalog["r_eff_pc_rmax_15pix_e+"][mask]

    return r_eff, r_eff_err_lo, r_eff_err_hi


def get_my_ages(catalog, mask):
    """
    Get the ages from my catalog, along with their errors

    :param catalog: Catalog to retrieve the ages from
    :param mask: Mask to apply to the data, to restrict to certain clusters
    :return: Tuple with three elements: age, lower age error, upper age error
    """
    age = catalog["age_yr"][mask]
    # age errors are reported as min and max values
    age_err_lo = age - catalog["age_yr_min"][mask]
    age_err_hi = catalog["age_yr_max"][mask] - age

    return age, age_err_lo, age_err_hi


def transform_to_log(mean, err_lo, err_hi):
    """
    Take a value and its error and transform this into the value and its error in log

    :param mean: Original value
    :param err_lo: Lower error bar
    :param err_hi: Upper error bar
    :return: log(mean), lower error in log, upper error in  log
    """
    log_mean = np.log10(mean)
    log_err_lo = log_mean - np.log10(mean - err_lo)
    log_err_hi = np.log10(mean + err_hi) - log_mean

    return log_mean, log_err_lo, log_err_hi
