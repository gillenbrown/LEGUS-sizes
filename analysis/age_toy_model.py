"""
age_toy_model.py - Create a toy model for the age evolution of the effective radius

This takes the following parameters:
- Path to save the plot
- Path to the file containing the best fit mass-radius relation split by age
"""

import sys
from pathlib import Path
import numpy as np
from scipy import optimize
from astropy import constants as c
from astropy import units as u

import betterplotlib as bpl

bpl.set_style()

# import some utils from the mass radius relation
mrr_dir = Path(__file__).resolve().parent / "mass_radius_relation"
sys.path.append(str(mrr_dir))
import mass_radius_utils_plotting as mru_p

# Get the input arguments
plot_name = Path(sys.argv[1])
fit_table_loc = Path(sys.argv[2])

# ======================================================================================
#
# load fit parameters for the age bins
#
# ======================================================================================
def is_fit_line(line):
    return "$\pm$" in line


def get_fit_from_line(line):
    quantities = line.split("&")
    # format nicer
    quantities = [q.strip() for q in quantities]
    name, N, beta, r_4, scatter, percentiles = quantities
    # get rid of the error, only include the quantity
    beta = float(beta.split()[0])
    r_4 = float(r_4.split()[0])
    scatter = float(scatter.split()[0])
    return name, beta, r_4


# then use these to find what we need
fits = dict()
with open(fit_table_loc, "r") as in_file:
    for line in in_file:
        if is_fit_line(line):
            name, beta, r_4 = get_fit_from_line(line)
            if "Age: " in name:
                if "1-10" in name:
                    name = "age1"
                elif "10-100" in name:
                    name = "age2"
                elif "100 Myr" in name:
                    name = "age3"
                else:
                    raise ValueError
                fits[name] = (beta, r_4)

# ======================================================================================
#
# Functions defining the relation as well as some simple evolution
#
# ======================================================================================
def mass_size_relation(mass, beta, r_4):
    return r_4 * (mass / 10 ** 4) ** beta


def stellar_mass_adiabatic(m_old, m_new, r_old):
    # Portegies Zwart et al. 2010 section 4.3.1 Eq 33
    # Krumholz review section 3.5.2
    r_new = r_old * m_old / m_new
    return r_new


def stellar_mass_rapid(m_old, m_new, r_old):
    # Portegies Zwart et al. 2010 section 4.2.1 Eq 31
    # Krumholz review section 3.5.2
    eta = m_new / m_old
    r_new = r_old * eta / (2 * eta - 1)
    return r_new


def tidal_radius(m_old, m_new, r_old):
    # Krumholz review Equation 9 section 1.3.2
    consts = r_old ** 3 / m_old
    r_new = (consts * m_new) ** (1 / 3)
    return r_new


# ======================================================================================
#
# Functions defining the evolution according to Gieles etal 2010
#
# ======================================================================================
# https://ui.adsabs.harvard.edu/abs/2010MNRAS.408L..16G/abstract
# Equation 6
def gieles_etal_10_evolution(initial_radius, initial_mass, time):
    # Equation 6 defines the main thing, but also grabs a few things from elsewhere
    m0 = initial_mass * u.Msun  # shorthand
    t = time * u.year  # shorthand
    r0 = initial_radius * u.pc

    delta = 0.07  # equation 4
    # equation 4 plus text after equation 7 for t_star. I don't have early ages so the
    # minimum doesn't matter to me here.
    t_star = 2e6 * u.year
    chi_t = 3 * (t / t_star) ** (-0.3)  # equation 7

    # equation 1 gives t_rh0
    print("fix column logarithm, look at citations")
    m_bar = 0.5 * u.Msun
    N = m0 / m_bar
    t_rh0 = 0.138 * np.sqrt(N * r0 ** 3 / (c.G * m_bar * np.log(0.1 * N) ** 2))

    # then fill out equation 6
    term_1 = (t / t_star) ** (2 * delta)
    term_2 = ((chi_t * t) / (t_rh0)) ** (4 / 3)
    r_final = r0 * np.sqrt(term_1 + term_2)

    # final mass given by equation 4
    m_final = m0 * (t / t_star) ** (-delta)
    return m_final.to("Msun").value, r_final.to("pc").value


# ======================================================================================
#
# Run clusters through this evolution
#
# ======================================================================================
mass_initial = np.logspace(3, 5, 100)
reff_observed_bin1 = mass_size_relation(mass_initial, *fits["age1"])
reff_observed_bin2 = mass_size_relation(mass_initial, *fits["age2"])
reff_observed_bin3 = mass_size_relation(mass_initial, *fits["age3"])

m_g10_30myr, r_g10_30myr = gieles_etal_10_evolution(
    reff_observed_bin1, mass_initial, 30e6
)
m_g10_300myr, r_g10_300myr = gieles_etal_10_evolution(
    reff_observed_bin1, mass_initial, 300e6
)

# ======================================================================================
#
# Simple fitting routine to get the parameters of resulting relations
#
# ======================================================================================
def negative_log_likelihood(params, xs, ys):
    # first convert the pivot point value into the intercept
    pivot_point_x = 4
    # so params[1] is really y(pivot_point_x) = m (pivot_point_x) + intercept
    intercept = params[1] - params[0] * pivot_point_x

    # calculate the difference
    data_diffs = ys - (params[0] * xs + intercept)

    # calculate the sum of data likelihoods. The total likelihood is the product of
    # individual cluster likelihoods, so when we take the log it turns into a sum of
    # individual log likelihoods.
    return np.sum(data_diffs ** 2)


def fit_mass_size_relation(mass, r_eff):
    log_mass = np.log10(mass)
    log_r_eff = np.log10(r_eff)

    # Then try the fitting
    best_fit_result = optimize.minimize(
        negative_log_likelihood,
        args=(
            log_mass,
            log_r_eff,
        ),
        bounds=([-1, 1], [None, None]),
        x0=np.array([0.2, np.log10(2)]),
    )
    assert best_fit_result.success
    beta = best_fit_result.x[0]
    r_4 = 10 ** best_fit_result.x[1]
    return beta, r_4


# ======================================================================================
#
# Make the plot
#
# ======================================================================================
def format_params(base_label, beta, r_4):
    return f"{base_label} - $\\beta={beta:.3f}, r_4={r_4:.3f}$"


colors = {
    "young": bpl.color_cycle[0],
    "med": bpl.color_cycle[5],
    "old": bpl.color_cycle[3],
}

fig, ax = bpl.subplots()
ax.plot(
    mass_initial,
    reff_observed_bin1,
    c=colors["young"],
    label=format_params("Age: 1-10 Myr Observed", *fits["age1"]),
)
ax.plot(
    mass_initial,
    reff_observed_bin2,
    c=colors["med"],
    label=format_params("Age: 10-100 Myr Observed", *fits["age2"]),
)
ax.plot(
    mass_initial,
    reff_observed_bin3,
    c=colors["old"],
    label=format_params("Age: 100 Myr - 1 Gyr Observed", *fits["age3"]),
)

ax.plot(
    m_g10_30myr,
    r_g10_30myr,
    c=colors["med"],
    ls="--",
    label=format_params(
        "Gieles+ 2010 - 30 Myr",
        *fit_mass_size_relation(m_g10_30myr, r_g10_30myr),
    ),
)

ax.plot(
    m_g10_300myr,
    r_g10_300myr,
    c=colors["old"],
    ls="--",
    label=format_params(
        "Gieles+ 2010 - 300 Myr",
        *fit_mass_size_relation(m_g10_300myr, r_g10_300myr),
    ),
)

mru_p.format_mass_size_plot(ax)
ax.legend(loc=3, fontsize=12)
fig.savefig(plot_name)
