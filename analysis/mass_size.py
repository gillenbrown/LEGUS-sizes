"""
mass_size.py - plot the mass-size relation for LEGUS clusters
"""
import sys
from pathlib import Path

import numpy as np
from astropy import table
from astropy.io import fits
from astropy import units as u
from astropy import constants as c
from scipy import optimize
import betterplotlib as bpl

# need to add the correct path to import utils
code_home_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(code_home_dir / "pipeline"))
import utils

bpl.set_style()

# ======================================================================================
#
# load the parameters the user passed in
#
# ======================================================================================
plot_name = Path(sys.argv[1])
oversampling_factor = int(sys.argv[2])
psf_width = int(sys.argv[3])
psf_source = sys.argv[4]
catalogs = [table.Table.read(item, format="ascii.ecsv") for item in sys.argv[5:]]
# then stack them together in one master catalog
big_catalog = table.vstack(catalogs, join_type="inner")

# then filter out some clusters
print(f"Total Clusters: {len(big_catalog)}")
mask = big_catalog["good"]

# mask = np.logical_and(mask, big_catalog["age_yr"] <= 200e6)
# print(f"Clusters with age < 200 Myr: {np.sum(mask)}")

mask = np.logical_and(mask, big_catalog["good"])
print(f"Clusters with good fits: {np.sum(mask)}")

mask = np.logical_and(mask, big_catalog["mass_msun"] > 0)
mask = np.logical_and(mask, big_catalog["mass_msun_max"] > 0)
mask = np.logical_and(mask, big_catalog["mass_msun_min"] > 0)
print(f"Clusters with good masses: {np.sum(mask)}")

mass_legus = big_catalog["mass_msun"][mask]
# mass errors are reported as min and max values
mass_err_hi_legus = big_catalog["mass_msun_max"][mask] - mass_legus
mass_err_lo_legus = mass_legus - big_catalog["mass_msun_min"][mask]

r_eff_legus = big_catalog["r_eff_pc_rmax_15pix_best"][mask]
r_eff_err_hi_legus = big_catalog["r_eff_pc_rmax_15pix_e+"][mask]
r_eff_err_lo_legus = big_catalog["r_eff_pc_rmax_15pix_e-"][mask]

# Then transform them into log
log_mass_legus = np.log10(mass_legus)
log_mass_err_hi_legus = np.log10(mass_legus + mass_err_hi_legus) - log_mass_legus
log_mass_err_lo_legus = log_mass_legus - np.log10(mass_legus - mass_err_lo_legus)

log_r_eff_legus = np.log10(r_eff_legus)
log_r_eff_err_hi_legus = np.log10(r_eff_legus + r_eff_err_hi_legus) - log_r_eff_legus
log_r_eff_err_lo_legus = log_r_eff_legus - np.log10(r_eff_legus - r_eff_err_lo_legus)


# ======================================================================================
#
# Load data from external sources - first is M31
#
# ======================================================================================
# M31 data. Masses and radii are in separate files.
data_path = code_home_dir / "analysis" / "krumholz_review_data"
johnson_12_table = fits.open(data_path / "johnson2012_m31.fits")
fouesneau_14_table = fits.open(data_path / "fouesneau2014_m31.fits")
# get the ids in both catalogs
johnson_12_id = johnson_12_table[1].data["PCID"]
fouesneau_14_id = fouesneau_14_table[1].data["PCID"]
m31_ids = np.intersect1d(johnson_12_id, fouesneau_14_id)

# get the relevant data
johnson_12_r_eff_arcsec = johnson_12_table[1].data["Reff"]
fouesneau_14_log_mass = fouesneau_14_table[1].data["logM-bset"]
fouesneau_14_log_mass_min = fouesneau_14_table[1].data["logM-p16"]
fouesneau_14_log_mass_max = fouesneau_14_table[1].data["logM-p84"]

# then restrict to ones that have ids that work
log_mass_m31 = []
log_mass_min_m31 = []
log_mass_max_m31 = []
r_eff_arcsec_m31 = []
for this_id in m31_ids:
    johnson_12_idx = np.where(johnson_12_id == this_id)[0]
    fouesneau_14_idx = np.where(fouesneau_14_id == this_id)[0]
    # numpy where gives arrays, we should only have one value here, make sure
    assert johnson_12_idx.size == 1
    assert fouesneau_14_idx.size == 1
    johnson_12_idx = johnson_12_idx[0]
    fouesneau_14_idx = fouesneau_14_idx[0]

    # check that there are no nans
    if np.isnan(fouesneau_14_log_mass[fouesneau_14_idx]) or np.isnan(
        johnson_12_r_eff_arcsec[fouesneau_14_idx]
    ):
        continue

    log_mass_m31.append(fouesneau_14_log_mass[fouesneau_14_idx])
    log_mass_min_m31.append(fouesneau_14_log_mass_min[fouesneau_14_idx])
    log_mass_max_m31.append(fouesneau_14_log_mass_max[fouesneau_14_idx])
    r_eff_arcsec_m31.append(johnson_12_r_eff_arcsec[fouesneau_14_idx])

log_mass_m31 = np.array(log_mass_m31)
log_mass_min_m31 = np.array(log_mass_min_m31)
log_mass_max_m31 = np.array(log_mass_max_m31)
r_eff_arcsec_m31 = np.array(r_eff_arcsec_m31)

# Johnson does not report errors on R_eff, so our only errors will be the distance
# errors
r_eff_full_m31 = utils.arcsec_to_pc_with_errors(Path("m31"), r_eff_arcsec_m31, 0, 0)
r_eff_m31, r_eff_err_lo_m31, r_eff_err_hi_m31 = r_eff_full_m31

# Then make both values have log and linear values for use in fitting and plotting
mass_m31 = 10 ** log_mass_m31
mass_err_lo_m31 = mass_m31 - 10 ** log_mass_min_m31
mass_err_hi_m31 = 10 ** log_mass_max_m31 - mass_m31

log_mass_err_lo_m31 = log_mass_m31 - log_mass_min_m31
log_mass_err_hi_m31 = log_mass_max_m31 - log_mass_m31

log_r_eff_m31 = np.log10(r_eff_m31)
log_r_eff_err_hi_m31 = np.log10(r_eff_m31 + r_eff_err_hi_m31) - log_r_eff_m31
log_r_eff_err_lo_m31 = log_r_eff_m31 - np.log10(r_eff_m31 - r_eff_err_lo_m31)

print(f"{len(mass_m31)} Clusters in M31")

# ======================================================================================
# Then the MW Open Clusters
# ======================================================================================
kharchenko_13_table = fits.open(data_path / "kharchenko2013_mw.fits")
kharchenko_13_mw_obj_type = kharchenko_13_table[1].data["Type"]
kharchenko_mw_dist = kharchenko_13_table[1].data["d"]
# restrict to solar neighborhood, not sure what the type does, but Krumholz uses it
kharchenko_good_idx = np.logical_and(
    [str(o_type) != "g" for o_type in kharchenko_13_mw_obj_type],
    kharchenko_mw_dist <= 2e3,
)
kharchenko_mw_Sigma = kharchenko_13_table[1].data["k"][kharchenko_good_idx]
kharchenko_mw_rt = kharchenko_13_table[1].data["rt"][kharchenko_good_idx]
kharchenko_mw_rc = kharchenko_13_table[1].data["rc"][kharchenko_good_idx]
kharchenko_mw_k = kharchenko_13_table[1].data["k"][kharchenko_good_idx]
kharchenko_mw_glat = kharchenko_13_table[1].data["GLAT"][kharchenko_good_idx]
kharchenko_mw_glon = kharchenko_13_table[1].data["GLON"][kharchenko_good_idx]
kharchenko_mw_dist = kharchenko_mw_dist[kharchenko_good_idx]

# Following code copied from Krumholz:
# Convert Kharchenko's King profile r_t and r_c measurements into
# projected half-mass / half-number radii and mass; mass is
# derived following equation (3) of Piskunov+ 2007, A&A, 468, 151,
# using updated values of the Oort constants from Bovy+ 2017, MNRAS,
# 468, L63, and the Sun's distance from the Galactic Center from
# Bland-Hawthorn & Gerhard, 2016, ARA&A, 54, 529; note that this
# calculation implicitly assumes that the Sun lies in the galactic
# plane, which is not exactly true (z0 ~= 25 pc), but the error
# associated with this approximation is small compared to the
# uncertainty in the distance to the Galctic Centre
kingtab = table.Table.read(data_path / "kingtab.txt", format="ascii")
kharchenko_logc = np.log10(kharchenko_mw_rt / kharchenko_mw_rc)
r_eff_mw_ocs = kharchenko_mw_rc * np.interp(
    kharchenko_logc, kingtab.columns["logc"], kingtab.columns["xh2d"]
)
oort_A0 = 15.3 * u.km / (u.s * u.kpc)
oort_B0 = -11.9 * u.km / (u.s * u.kpc)
R0 = 8.2 * u.kpc
mw_rgc = np.sqrt(
    R0 ** 2
    + (kharchenko_mw_dist * u.pc) ** 2
    - 2.0 * R0 * kharchenko_mw_dist * u.pc * np.cos(kharchenko_mw_glon * np.pi / 180)
)
drg = (mw_rgc - R0) / R0
oort_A = oort_A0 * (1.0 - drg)
oort_A_minus_B = oort_A0 - oort_B0 - 2.0 * oort_A0 * drg
mass_mw_ocs = (
    (4.0 * oort_A * oort_A_minus_B * (kharchenko_mw_rt * u.pc) ** 3 / c.G)
    .to("Msun")
    .value
)

# Errors are not present.
r_eff_err_mw_ocs = np.zeros(r_eff_mw_ocs.shape)
log_r_eff_err_mw_ocs = np.zeros(r_eff_mw_ocs.shape)
mass_err_mw_ocs = np.zeros(r_eff_mw_ocs.shape)
log_mass_err_mw_ocs = np.zeros(r_eff_mw_ocs.shape)

# then convert this to linear space for plotting
log_r_eff_mw_ocs = np.log10(r_eff_mw_ocs)
log_mass_mw_ocs = np.log10(mass_mw_ocs)

print(f"{len(mass_mw_ocs)} Open Clusters in MW")

# ======================================================================================
# Then the MW Globular Clusters
# ======================================================================================
# Read Baumgardt & Hilker 2018 table for MW globulars
mw_gc_table = table.Table.read(
    data_path / "baumgardt2018.txt",
    format="ascii.fixed_width",
    col_starts=[
        0,
        9,
        20,
        31,
        38,
        46,
        52,
        61,
        70,
        81,
        92,
        100,
        107,
        114,
        121,
        128,
        136,
        145,
        153,
        161,
        169,
        176,
        183,
        190,
        198,
    ],
    header_start=0,
    data_start=2,
)
mass_mw_gcs = mw_gc_table["Mass"]
mass_err_mw_gcs = mw_gc_table["DM"]
r_eff_mw_gcs = mw_gc_table["rh,l"]
# no r_eff errors
r_eff_err_mw_gcs = np.zeros(r_eff_mw_gcs.shape)
log_r_eff_err_mw_gcs = np.zeros(r_eff_mw_gcs.shape)

# convert to log
log_mass_mw_gcs = np.log10(mass_mw_gcs)
log_r_eff_mw_gcs = np.log10(r_eff_mw_gcs)
log_mass_err_lo_mw_gcs = log_mass_mw_gcs - np.log10(mass_mw_gcs - mass_err_mw_gcs)
log_mass_err_hi_mw_gcs = np.log10(mass_mw_gcs + mass_err_mw_gcs) - log_mass_mw_gcs

print(f"{len(mass_mw_gcs)} Globular Clusters in MW")

# ======================================================================================
# Then the M31 Globular Clusters
# ======================================================================================
# Read Barmby+ 2007 table for M31 globulars
# m31_gc_tab = table.Table.read(data_path / "barmby2007_m31.txt", format="ascii.fixed_width")
# m31_gc_m = 10.**m31_gc_tab.columns['logM']
# m31_gc_Sigmah = 10.**m31_gc_tab.columns['logSh']
# m31_gc_rh = np.sqrt(m31_gc_m/m31_gc_Sigmah)

# ======================================================================================
#
# Fit the mass-size model
#
# ======================================================================================
# This fitting is based off the prescriptions in Hogg, Bovy, Lang 2010 (arxiv:1008.4686)
# sections 7 and 8. We incorporate the uncertainties in x and y by calculating the
# difference and sigma along the line orthogonal to the best fit line.

# define the functions that project the distance and uncertainty along the line
def unit_vector_perp_to_line(slope):
    """
    Equation 29 of Hogg, Bovy, Lang 2010 (arxiv:1008.4686)
    :param slope: Slope of the line
    :return: Two component unit vector perpendicular to the line
    """
    return np.array([-slope, 1]).T / np.sqrt(1 + slope ** 2)


def project_data_differences(xs, ys, slope, intercept):
    """
    Calculate the orthogonal displacement of all data points from the line specified.
    See equation 30 of Hogg, Bovy, Lang 2010 (arxiv:1008.4686)

    I made a simpler version of this equation by examining the geometry of the
    situation. The orthogonal direction to a line can be obtained fairly easily.

    I include a commented out version of the original implementation. Both
    implementations give the same results

    :param slope: Slope of the line
    :param intercept: Intercept of the line
    :return: Orthogonal displacement of all datapoints from this line
    """
    # v_hat = unit_vector_perp_to_line(slope)
    # data_array = np.array([masses_log, r_eff_log])
    # dot_term = np.dot(v_hat, data_array)
    #
    # theta = np.arctan(slope)
    # return dot_term - intercept * np.cos(theta)

    return np.cos(np.arctan(slope)) * (ys - (slope * xs + intercept))


def project_data_variance(
    xs, x_err_down, x_err_up, ys, y_err_down, y_err_up, slope, intercept
):
    """
    Calculate the orthogonal uncertainty of all data points from the line specified.
    See equation 31 of Hogg, Bovy, Lang 2010 (arxiv:1008.4686)

    :param slope: Slope of the line
    :param intercept: Intercept of the line
    :return: Orthogonal displacement of all datapoints from this line
    """
    # make dummy error arrays, which will be filled later
    x_errors = np.zeros(xs.shape)
    y_errors = np.zeros(ys.shape)
    # determine which errors to use. This is done on a datapoint by datapoint basis.
    # If the datapoint is above the best fit line, we use the lower errors on y,
    # if it is below the line we use the upper errors.
    expected_values = xs * slope + intercept
    mask_above = ys > expected_values

    y_errors[mask_above] = y_err_down[mask_above]
    y_errors[~mask_above] = y_err_up[~mask_above]

    # Errors on x are similar, but it depends on the sign of the slope. For a
    # positive slope, we use the upper errors for points above the line, and lower
    # errors for points below the line. This is opposite for negative slope. This can
    # be determined by examining the direction the orthogonal line will go in each of
    # these cases.
    if slope > 0:
        x_errors[mask_above] = x_err_up[mask_above]
        x_errors[~mask_above] = x_err_down[~mask_above]
    else:
        x_errors[mask_above] = x_err_down[mask_above]
        x_errors[~mask_above] = x_err_up[~mask_above]
    # convert to variance
    x_variance = x_errors ** 2
    y_variance = y_errors ** 2

    # Then we follow the equation 31 to project this along the direction requested.
    # Since our covariance array is diagonal already (no covariance terms), Equation
    # 26 is simple and Equation 31 can be simplified. Note that this has limits
    # of x_variance if slope = infinity (makes sense, as the horizontal direction
    # would be perpendicular to that line), and y_variance if slope = 0 (makes
    # sense, as the vertical direction is perpendicular to that line).
    return (slope ** 2 * x_variance + y_variance) / (1 + slope ** 2)


# Then we can define the functions to minimize
def negative_log_likelihood(params, xs, x_err_down, x_err_up, ys, y_err_down, y_err_up):
    """
    Function to be minimized. We use negative log likelihood, as minimizing this
    maximizes likelihood.

    The functional form is taken from Hogg, Bovy, Lang 2010 (arxiv:1008.4686) eq 35.

    :param params: Slope, height at the pivot point, and standard deviation of
                   intrinsic scatter
    :return: Value for the negative log likelihood
    """
    # first convert the pivot point value into the intercept
    pivot_point_x = 4
    # so params[1] is really y(pivot_point_x) = m (pivot_point_x) + intercept
    intercept = params[1] - params[0] * pivot_point_x

    data_variance = project_data_variance(
        xs, x_err_down, x_err_up, ys, y_err_down, y_err_up, params[0], intercept
    )
    data_diffs = project_data_differences(xs, ys, params[0], intercept)

    # calculate the sum of data likelihoods
    data_likelihoods = -0.5 * np.sum(
        (data_diffs ** 2) / (data_variance + params[2] ** 2)
    )
    # then penalize large intrinsic scatter. This term really comes from the definition
    # of a Gaussian likelihood. This term is always out front of a Gaussian, but
    # normally it's just a constant. When we include intrinsic scatter it now
    # affects the likelihood.
    scatter_likelihood = -0.5 * np.sum(np.log(data_variance + params[2] ** 2))
    # up to a constant, the sum of these is the likelihood. Return the negative of it
    # to get the negative log likelihood
    return -1 * (data_likelihoods + scatter_likelihood)


def fit_mass_size_relation(
    log_mass,
    log_mass_err_lo,
    log_mass_err_hi,
    log_r_eff,
    log_r_eff_err_lo,
    log_r_eff_err_hi,
    fit_mass_lower_limit=1e-5,
    fit_mass_upper_limit=1e10,
):
    fit_mask = log_mass > np.log10(fit_mass_lower_limit)
    fit_mask = np.logical_and(fit_mask, log_mass < np.log10(fit_mass_upper_limit))

    log_mass = log_mass[fit_mask]
    log_mass_err_lo = log_mass_err_lo[fit_mask]
    log_mass_err_hi = log_mass_err_hi[fit_mask]
    log_r_eff = log_r_eff[fit_mask]
    log_r_eff_err_lo = log_r_eff_err_lo[fit_mask]
    log_r_eff_err_hi = log_r_eff_err_hi[fit_mask]

    # set some of the convergence criteria parameters for the Powell fitting routine.
    xtol = 1e-10
    ftol = 1e-10
    maxfev = np.inf
    maxiter = np.inf
    # Then try the fitting
    best_fit_result = optimize.minimize(
        negative_log_likelihood,
        args=(
            log_mass,
            log_mass_err_lo,
            log_mass_err_hi,
            log_r_eff,
            log_r_eff_err_lo,
            log_r_eff_err_hi,
        ),
        bounds=([-1, 1], [None, None], [0, 0.5]),
        x0=np.array([0.2, np.log10(2), 0.3]),
        method="Powell",
        options={
            "xtol": xtol,
            "ftol": ftol,
            "maxfev": maxfev,
            "maxiter": maxiter,
        },
    )
    assert best_fit_result.success

    # best_slope, best_intercept, best_intrinsic_scatter = best_fit_result.x

    # Then do bootstrapping
    n_variables = len(best_fit_result.x)
    param_history = [[] for _ in range(n_variables)]
    param_std_last = [np.inf for _ in range(n_variables)]

    converge_criteria = 0.01  # fractional change in std required for convergence
    converged = [False for _ in range(n_variables)]
    check_spacing = 20  # how many iterations between checking the std
    iteration = 0
    while not all(converged):
        iteration += 1

        # create a new sample of x and y coordinates
        sample_idxs = np.random.randint(0, len(log_mass), len(log_mass))

        # fit to this set of data
        this_result = optimize.minimize(
            negative_log_likelihood,
            args=(
                log_mass[sample_idxs],
                log_mass_err_lo[sample_idxs],
                log_mass_err_hi[sample_idxs],
                log_r_eff[sample_idxs],
                log_r_eff_err_lo[sample_idxs],
                log_r_eff_err_hi[sample_idxs],
            ),
            bounds=([-1, 1], [None, None], [0, 0.5]),
            x0=best_fit_result.x,
            method="Powell",
            options={
                "xtol": xtol,
                "ftol": ftol,
                "maxfev": maxfev,
                "maxiter": maxiter,
            },
        )

        assert this_result.success
        # store the results
        for param_idx in range(n_variables):
            param_history[param_idx].append(this_result.x[param_idx])

        # then check if we're converged
        if iteration % check_spacing == 0:
            print(f"Bootstrap Iteration {iteration}")
            for param_idx in range(n_variables):
                # calculate the new standard deviation
                this_std = np.std(param_history[param_idx])
                if this_std == 0:
                    converged[param_idx] = True
                else:  # actually calculate the change
                    last_std = param_std_last[param_idx]
                    diff = abs((this_std - last_std) / this_std)
                    converged[param_idx] = diff < converge_criteria

                # then set the new last value
                param_std_last[param_idx] = this_std

    return best_fit_result.x, param_history


# ======================================================================================
#
# make the plot
#
# ======================================================================================
def get_r_percentiles(radii, masses, percentile, d_log_M):
    bins = np.logspace(0, 7, int(5 / d_log_M) + 1)

    bin_centers = []
    radii_percentiles = []
    for idx in range(len(bins) - 1):
        lower = bins[idx]
        upper = bins[idx + 1]

        # then find all clusters in this mass range
        mask_above = masses > lower
        mask_below = masses < upper
        mask_good = np.logical_and(mask_above, mask_below)

        good_radii = radii[mask_good]
        print(lower, upper, len(good_radii))
        if len(good_radii) > 20:
            radii_percentiles.append(np.percentile(good_radii, percentile))
            # the bin centers will be the mean in log space
            bin_centers.append(10 ** np.mean([np.log10(lower), np.log10(upper)]))

    return bin_centers, radii_percentiles


def get_r_percentiles_moving(radii, masses, percentile, n, dn):
    # go through the masses in sorted order
    idxs_sort = np.argsort(masses)
    # then go through chunks of them at a time to get the medians
    masses_median = []
    radii_percentiles = []
    for left_idx in range(0, len(radii) - dn, dn):
        right_idx = left_idx + n
        # fix the last bin
        if right_idx > len(idxs_sort):
            right_idx = len(idxs_sort)
            left_idx = right_idx - n

        idxs = idxs_sort[left_idx:right_idx]
        this_masses = masses[idxs]
        this_radii = radii[idxs]

        masses_median.append(np.median(this_masses))
        radii_percentiles.append(np.percentile(this_radii, percentile))
    return masses_median, radii_percentiles


def get_r_percentiles_unique_values(radii, ages, percentile):
    # get the unique ages
    unique_ages = np.unique(ages)
    # cut off values above 1e9
    unique_ages = unique_ages[unique_ages <= 1e9]
    radii_percentiles = []
    for age in unique_ages:
        mask = ages == age
        radii_percentiles.append(np.percentile(radii[mask], percentile))
    return unique_ages, radii_percentiles


def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def measure_psf_reff(psf):
    # the center is the central pixel of the image
    x_cen = int((psf.shape[1] - 1.0) / 2.0)
    y_cen = int((psf.shape[0] - 1.0) / 2.0)
    total = np.sum(psf)
    half_light = total / 2.0
    # then go through all the pixel values to determine the distance from the center.
    # Then we can go through them in order to determine the half mass radius
    radii = []
    values = []
    for x in range(psf.shape[1]):
        for y in range(psf.shape[1]):
            # need to include the oversampling factor in the distance
            radii.append(distance(x, y, x_cen, y_cen) / oversampling_factor)
            values.append(psf[y][x])

    idxs_sort = np.argsort(radii)
    sorted_radii = np.array(radii)[idxs_sort]
    sorted_values = np.array(values)[idxs_sort]

    cumulative_light = 0
    for idx in range(len(sorted_radii)):
        cumulative_light += sorted_values[idx]
        if cumulative_light >= half_light:
            return sorted_radii[idx]


def plot_best_fit_line(
    ax, best_fit_params, fit_mass_lower_limit=1, fit_mass_upper_limit=1e6
):
    # first convert the pivot point value into the intercept
    pivot_point_x = 4
    # so params[1] is really y(pivot_point_x) = m (pivot_point_x) + intercept
    intercept = best_fit_params[1] - best_fit_params[0] * pivot_point_x

    plot_log_masses = np.arange(
        np.log10(fit_mass_lower_limit), np.log10(fit_mass_upper_limit), 0.01
    )
    plot_log_radii = best_fit_params[0] * plot_log_masses + intercept
    ax.plot(
        10 ** plot_log_masses,
        10 ** plot_log_radii,
        c=bpl.color_cycle[1],
        lw=4,
        zorder=10,
        label="$R_{eff} \propto M^{" + f"{best_fit_params[0]:.2f}" + "}$",
    )
    ax.fill_between(
        x=10 ** plot_log_masses,
        y1=10 ** (plot_log_radii - best_fit_params[2]),
        y2=10 ** (plot_log_radii + best_fit_params[2]),
        color="0.9",
        zorder=0,
        label="$\sigma_{int}$" + f" = {best_fit_params[2]:.2f} dex",
    )

    # Filled in bootstrap interval is currently turned off because the itnerval is smaller
    # than the width of the line
    # # Then add the shaded region of regions allowed by bootstrapping. We'll calculate
    # # the fit line for all the iterations, then at each x value calculate the 68
    # # percent range to shade between.
    # lines = [[] for _ in range(len(plot_log_masses))]
    # for i in range(len(param_history[0])):
    #     this_line = param_history[0][i] * plot_log_masses + param_history[1][i]
    #     for j in range(len(this_line)):
    #         lines[j].append(this_line[j])
    # # Then we can calculate the percentile at each location. The y is in log here,
    # # so scale it back up to regular values
    # upper_limits = [10 ** np.percentile(ys, 84.15) for ys in lines]
    # lower_limits = [10 ** np.percentile(ys, 15.85) for ys in lines]
    #
    # ax.fill_between(
    #     x=10 ** plot_log_masses,
    #     y1=lower_limits,
    #     y2=upper_limits,
    #     zorder=0,
    #     alpha=0.5,
    # )


def add_psfs_to_plot(ax, x_max=1e6):
    # then add all the PSF widths. Here we load the PSF and directly measure it's R_eff,
    # so we can have a fair comparison to the clusters
    for cat_loc in sys.argv[5:]:
        size_home_dir = Path(cat_loc).parent
        home_dir = size_home_dir.parent

        psf_name = (
            f"psf_"
            f"{psf_source}_stars_"
            f"{psf_width}_pixels_"
            f"{oversampling_factor}x_oversampled.fits"
        )

        psf = fits.open(size_home_dir / psf_name)["PRIMARY"].data
        psf_size_pixels = measure_psf_reff(psf)
        psf_size_arcsec = utils.pixels_to_arcsec(psf_size_pixels, home_dir)
        psf_size_pc = utils.arcsec_to_pc_with_errors(
            home_dir, psf_size_arcsec, 0, 0, False
        )[0]
        ax.plot(
            [0.7 * x_max, x_max],
            [psf_size_pc, psf_size_pc],
            lw=1,
            c=bpl.almost_black,
            zorder=3,
        )


def format_mass_size_plot(ax, xmin=1e2, xmax=1e6):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_limits(xmin, xmax, 0.1, 40)
    ax.add_labels("Cluster Mass [M$_\odot$]", "Cluster Effective Radius [pc]")
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.legend(loc=2, frameon=False)


def plot_mass_size_dataset(
    mass, mass_err_lo, mass_err_hi, r_eff, r_eff_err_lo, r_eff_err_hi, color, label=None
):
    ax.scatter(mass, r_eff, alpha=1.0, s=3, c=color, zorder=4, label=label)
    # Have errorbars separately so they can easily be turned off
    ax.errorbar(
        x=mass,
        y=r_eff,
        alpha=1.0,
        markersize=0,
        yerr=[r_eff_err_lo, r_eff_err_hi],
        xerr=[mass_err_lo, mass_err_hi],
        lw=0.1,
        zorder=3,
        c=color,
    )


def add_percentile_lines(ax, mass, r_eff, style="moving"):
    # plot the median and the IQR
    for percentile in [5, 25, 75, 95]:
        if style == "moving":
            mass_bins, radii_percentile = get_r_percentiles_moving(
                r_eff, mass, percentile, 200, 200
            )
        elif style == "unique":
            mass_bins, radii_percentile = get_r_percentiles_unique_values(
                r_eff, mass, percentile
            )
        elif style == "fixed_width":
            mass_bins, radii_percentile = get_r_percentiles(
                r_eff, mass, percentile, 0.1
            )
        else:
            raise ValueError("Style not recognized")
        ax.plot(
            mass_bins,
            radii_percentile,
            c=bpl.almost_black,
            lw=3 * (1 - (abs(percentile - 50) / 50)) + 0.5,
            zorder=9,
        )
        ax.text(
            x=mass_bins[0],
            y=radii_percentile[0],
            ha="center",
            va="bottom",
            s=percentile,
            fontsize=16,
        )


# ======================================================================================
#
# Then actually make different versions of this plot
#
# ======================================================================================
fit_legus, fit_legus_history = fit_mass_size_relation(
    log_mass_legus,
    log_mass_err_lo_legus,
    log_mass_err_hi_legus,
    log_r_eff_legus,
    log_r_eff_err_lo_legus,
    log_r_eff_err_hi_legus,
    fit_mass_lower_limit=1e2,
    fit_mass_upper_limit=1e5,
)

fig, ax = bpl.subplots()
plot_mass_size_dataset(
    mass_legus,
    mass_err_lo_legus,
    mass_err_hi_legus,
    r_eff_legus,
    r_eff_err_lo_legus,
    r_eff_err_hi_legus,
    bpl.color_cycle[0],
)
add_percentile_lines(ax, mass_legus, r_eff_legus)
plot_best_fit_line(ax, fit_legus, 1e2, 1e5)
# add_psfs_to_plot(ax)
format_mass_size_plot(ax)
fig.savefig(plot_name)

# Then have another plot with many datasets
fit_combo, fit_legus_combo = fit_mass_size_relation(
    np.concatenate([log_mass_legus, log_mass_m31, log_mass_mw_ocs]),
    np.concatenate(
        [
            log_mass_err_lo_legus,
            log_mass_err_lo_m31,
            log_mass_err_mw_ocs,
        ]
    ),
    np.concatenate(
        [
            log_mass_err_hi_legus,
            log_mass_err_hi_m31,
            log_mass_err_mw_ocs,
        ]
    ),
    np.concatenate([log_r_eff_legus, log_r_eff_m31, log_r_eff_mw_ocs]),
    np.concatenate(
        [
            log_r_eff_err_lo_legus,
            log_r_eff_err_lo_m31,
            log_r_eff_err_mw_ocs,
        ]
    ),
    np.concatenate(
        [
            log_r_eff_err_hi_legus,
            log_r_eff_err_hi_m31,
            log_r_eff_err_mw_ocs,
        ]
    ),
    fit_mass_lower_limit=1,
    fit_mass_upper_limit=1e5,
)
fig, ax = bpl.subplots()
plot_mass_size_dataset(
    mass_legus,
    mass_err_lo_legus,
    mass_err_hi_legus,
    r_eff_legus,
    r_eff_err_lo_legus,
    r_eff_err_hi_legus,
    bpl.color_cycle[0],
    "LEGUS",
)
plot_mass_size_dataset(
    mass_m31,
    mass_err_lo_m31,
    mass_err_hi_m31,
    r_eff_m31,
    r_eff_err_lo_m31,
    r_eff_err_hi_m31,
    bpl.color_cycle[3],
    "M31",
)
plot_mass_size_dataset(
    mass_mw_ocs,
    mass_err_mw_ocs,
    mass_err_mw_ocs,
    r_eff_mw_ocs,
    r_eff_err_mw_ocs,
    r_eff_err_mw_ocs,
    bpl.color_cycle[4],
    "MW Open Clusters",
)

add_percentile_lines(
    ax,
    np.concatenate([mass_legus, mass_m31, mass_mw_ocs]),
    np.concatenate([r_eff_legus, r_eff_m31, r_eff_mw_ocs]),
)
# add_percentile_lines(ax, mass_m31, r_eff_m31, style="unique")
plot_best_fit_line(ax, fit_combo, 1, 1e5)
# add_psfs_to_plot(ax, x_max=1e7)
format_mass_size_plot(ax, xmin=1, xmax=1e7)
fig.savefig(plot_name.parent / "mass_size_combo.png")

# ======================================================================================
#
# similar plot
#
# ======================================================================================
fig, ax = bpl.subplots()

# then add all the PSF widths. Here we load the PSF and directly measure it's R_eff,
# so we can have a fair comparison to the clusters
all_ratios = np.array([])
for cat_loc in sys.argv[5:]:
    cat = table.Table.read(cat_loc, format="ascii.ecsv")

    size_home_dir = Path(cat_loc).parent
    home_dir = size_home_dir.parent

    psf_name = (
        f"psf_"
        f"{psf_source}_stars_"
        f"{psf_width}_pixels_"
        f"{oversampling_factor}x_oversampled.fits"
    )

    psf = fits.open(size_home_dir / psf_name)["PRIMARY"].data
    psf_size = measure_psf_reff(psf)

    this_ratio = cat[f"r_eff_pixels_rmax_15pix_best"].data / psf_size
    all_ratios = np.concatenate([all_ratios, this_ratio])

ax.hist(
    all_ratios,
    alpha=1.0,
    lw=1,
    color=bpl.color_cycle[3],
    bins=np.logspace(-1, 1, 21),
)

ax.axvline(1.0)
ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_limits(0.1, 10)
ax.add_labels("Cluster Effective Radius / PSF Effective Radius", "Number of Clusters")
ax.xaxis.set_ticks_position("both")
ax.yaxis.set_ticks_position("both")
fig.savefig(plot_name.parent / "r_eff_over_psf.png", bbox_inches="tight")
