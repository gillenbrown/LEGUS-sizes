"""
density.py - Create a plot showing the density of clusters.

This takes the following parameters:
- Path to save the plot
- Then the paths to all the final catalogs.
"""

import sys
from pathlib import Path
from astropy import table
import numpy as np
from scipy import optimize
from matplotlib import ticker
import betterplotlib as bpl

bpl.set_style()

# import a colormap function from the mass-size relation plots
code_home_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(code_home_dir / "analysis" / "mass_radius_relation"))
from mass_radius_utils_plotting import create_color_cmap

# ======================================================================================
#
# Load the catalogs that were passed in
#
# ======================================================================================
plot_name = Path(sys.argv[1]).resolve()
fits_output_name = Path(sys.argv[2]).resolve()

catalogs = [table.Table.read(item, format="ascii.ecsv") for item in sys.argv[3:]]
# then stack them together in one master catalog
big_catalog = table.vstack(catalogs, join_type="inner")

# restrict to clusters with good masses and radii
good_mask = np.logical_and(big_catalog["good_radius"], big_catalog["good_fit"])
big_catalog = big_catalog[good_mask]

# ======================================================================================
#
# Get the quantities we'll need for the plot
#
# ======================================================================================
density_3d = big_catalog["3d_density"]
density_3d_log_err = big_catalog["3d_density_log_err"]
density_2d = big_catalog["surface_density"]
density_2d_log_err = big_catalog["surface_density_log_err"]

# turn these errors into linear space for plotting
density_3d_err_lo = density_3d - 10 ** (np.log10(density_3d) - density_3d_log_err)
density_3d_err_hi = 10 ** (np.log10(density_3d) + density_3d_log_err) - density_3d

density_2d_err_lo = density_2d - 10 ** (np.log10(density_2d) - density_2d_log_err)
density_2d_err_hi = 10 ** (np.log10(density_2d) + density_2d_log_err) - density_2d

# then mass
mass = big_catalog["mass_msun"]
m_err_lo = big_catalog["mass_msun"] - big_catalog["mass_msun"]
m_err_hi = big_catalog["mass_msun_max"] - big_catalog["mass_msun"]

# also set up the masks for age
age = big_catalog["age_yr"]
mask_young = age < 1e7
mask_med = np.logical_and(age >= 1e7, age < 1e8)
mask_old = np.logical_and(age >= 1e8, age < 1e9)
mask_all = age < np.inf


# ======================================================================================
#
# Convenience functions
#
# ======================================================================================
def gaussian(x, mean, variance):
    """
    Normalized Gaussian Function at a given value.

    Is normalized to integrate to 1.

    :param x: value to calculate the Gaussian at
    :param mean: mean value of the Gaussian
    :param variance: Variance of the Gaussian distribution
    :return: log of the likelihood at x
    """
    exp_term = np.exp(-((x - mean) ** 2) / (2 * variance))
    normalization = 1.0 / np.sqrt(2 * np.pi * variance)
    return exp_term * normalization


def fit_gaussian(x_data, y_data):
    def to_minimize(params):
        norm, mean, variance = params
        predicted = norm * gaussian(x_data, mean, variance)
        return np.sum((predicted - y_data) ** 2)

    fit = optimize.minimize(
        to_minimize,
        x0=(1, 1, 1),
        bounds=((None, None), (None, None), (0.001, None)),  # variance is positive
    )
    assert fit.success
    return fit.x


def kde(x_grid, log_x, log_x_err):
    ys = np.zeros(x_grid.size)
    log_x_grid = np.log10(x_grid)

    for lx, lxe in zip(log_x, log_x_err):
        ys += gaussian(log_x_grid, lx, lxe ** 2)

    # # normalize the y value
    ys = np.array(ys)
    ys = 200 * ys / np.sum(ys)  # arbitrary scaling to look nice
    return ys


def contour(ax, mass, r_eff, color, zorder):
    cmap = create_color_cmap(color, 0.1, 0.8)
    common = {
        "percent_levels": [0.5, 0.90],
        "smoothing": [0.15, 0.15],  # dex
        "bin_size": 0.02,  # dex
        "log": True,
        "cmap": cmap,
    }
    ax.density_contourf(mass, r_eff, alpha=0.25, zorder=zorder, **common)
    ax.density_contour(mass, r_eff, zorder=zorder + 1, **common)


# Function to use to set the ticks
@ticker.FuncFormatter
def nice_log_formatter(x, pos):
    exp = np.log10(x)
    # this only works for labels that are factors of 10. Other values will produce
    # misleading results, so check this assumption.
    assert np.isclose(exp, int(exp))

    # for values between 0.01 and 100, just use that value.
    # Otherwise use the log.
    if abs(exp) < 2:
        return f"{x:g}"
    else:
        return f"$10^{exp:.0f}$"


# ======================================================================================
#
# Start the table to output the fit parameters to
#
# ======================================================================================
out_file = open(fits_output_name, "w")
# write the header
out_file.write("\t\\begin{tabular}{lcccc}\n")
out_file.write("\t\t\\toprule\n")
out_file.write(
    "\t\tAge & "
    "$\log \mu_\\rho$ & "
    "$\sigma_\\rho$ (dex) & "
    "$\log \mu_\Sigma$ & "
    "$\sigma_\Sigma$ (dex) \\\\ \n"
)
out_file.write("\t\t\midrule\n")


def write_fit_line(name, mean_2d, std_2d, mean_3d, std_3d):
    out_file.write(
        f"\t\t{name.replace('-', '--').replace('Age: ', '')} & "
        f"{mean_3d:.2f} & "
        f"{std_3d:.2f} & "
        f"{mean_2d:.2f} & "
        f"{std_2d:.2f} \\\\ \n"
    )


# ======================================================================================
#
# Make the plot
#
# ======================================================================================
fig, axs = bpl.subplots(figsize=[13, 13], ncols=2, nrows=2)
ax_3_k = axs[0][0]
ax_3_m = axs[1][0]
ax_2_k = axs[0][1]
ax_2_m = axs[1][1]

density_grid = np.logspace(-2, 6, 1000)

for mask, name, color, zorder in zip(
    [mask_all, mask_young, mask_med, mask_old],
    ["All", "Age: 1-10 Myr", "Age: 10-100 Myr", "Age: 100 Myr - 1 Gyr"],
    [None, bpl.color_cycle[0], bpl.color_cycle[5], bpl.color_cycle[3]],
    [None, 3, 2, 1],
):
    # create the KDE histogram for the top panel
    kde_2d = kde(density_grid, np.log10(density_2d[mask]), density_2d_log_err[mask])
    kde_3d = kde(density_grid, np.log10(density_3d[mask]), density_3d_log_err[mask])

    # fit this with a Gaussian
    norm_2d, mean_2d, variance_2d = fit_gaussian(np.log10(density_grid), kde_2d)
    norm_3d, mean_3d, variance_3d = fit_gaussian(np.log10(density_grid), kde_3d)

    write_fit_line(name, mean_2d, np.sqrt(variance_2d), mean_3d, np.sqrt(variance_3d))

    # plotting doesn't happen for all subsets. Set None as the color to skip plotting
    if color is None:
        continue

    # plot the KDE histograms
    ax_3_k.plot(density_grid, kde_3d, c=color)
    ax_2_k.plot(density_grid, kde_2d, c=color, label=name)

    # # then plot the fit to those histograms
    # plot_fit_2d = norm_2d * gaussian(np.log10(density_grid), mean_2d, variance_2d)
    # plot_fit_3d = norm_3d * gaussian(np.log10(density_grid), mean_3d, variance_3d)
    # ax_2_k.plot(density_grid, plot_fit_2d, ls=":", c=color)
    # ax_3_k.plot(density_grid, plot_fit_3d, ls=":", c=color)

    # plot the contours in the lower panels
    contour(ax_3_m, mass[mask], density_3d[mask], color, zorder)
    contour(ax_2_m, mass[mask], density_2d[mask], color, zorder)

# format axes
ax_2_k.legend(loc=2, fontsize=14, frameon=False)
for ax in axs.flatten():
    ax.set_xscale("log")
    ax.tick_params(axis="both", which="major", length=8)
    ax.tick_params(axis="both", which="minor", length=4)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_major_formatter(nice_log_formatter)
for ax in axs[0]:
    ax.set_limits(0.1, 1e5, 0)
for ax in axs[1]:
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(nice_log_formatter)
    ax.set_limits(1e2, 1e6, 0.1, 1e5)

# add labels to the axes
label_mass = "Mass [$M_\odot$]"
label_kde_2d = "Normalized dN/dlog($\\Sigma_h$)"
label_kde_3d = "Normalized dN/dlog($\\rho_h$)"
label_3d = "Density [$M_\odot$/pc$^3$]"
label_2d = "Surface Density [$M_\odot$/pc$^2$]"
ax_3_k.add_labels(label_3d, label_kde_3d)
ax_3_m.add_labels(label_mass, label_3d)
ax_2_k.add_labels(label_2d, label_kde_2d)
ax_2_m.add_labels(label_mass, label_2d)

fig.savefig(plot_name)

# then finalize the output file
out_file.write("\t\t\\bottomrule\n")
out_file.write("\t\end{tabular}\n")
out_file.close()
