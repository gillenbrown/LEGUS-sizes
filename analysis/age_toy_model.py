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
    return r_4 * u.pc * (mass / (10 ** 4 * u.Msun)) ** beta


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
# Equation 6 is the key
# this includes stellar evolution and two-body relaxation. No tidal evolution
def gieles_etal_10_evolution(initial_radius, initial_mass, time):
    # all quantities must have units
    # Equation 6 defines the main thing, but also grabs a few things from elsewhere
    m0 = initial_mass  # shorthand
    t = time  # shorthand
    r0 = initial_radius

    delta = 0.07  # equation 4
    # equation 4 plus text after equation 7 for t_star. I don't have early ages so the
    # minimum doesn't matter to me here.
    t_star = 2e6 * u.year
    chi_t = 3 * (t / t_star) ** (-0.3)  # equation 7

    # equation 1 gives t_rh0
    print("fix Coulomb logarithm, look at citations")
    m_bar = 0.5 * u.Msun
    N = m0 / m_bar
    t_rh0 = 0.138 * np.sqrt(N * r0 ** 3 / (c.G * m_bar * np.log(0.4 * N) ** 2))

    # then fill out equation 6
    term_1 = (t / t_star) ** (2 * delta)
    term_2 = ((chi_t * t) / (t_rh0)) ** (4 / 3)
    r_final = r0 * np.sqrt(term_1 + term_2)

    # final mass given by equation 4
    m_final = m0 * (t / t_star) ** (-delta)
    return m_final.to(u.Msun), r_final.to(u.pc)


# ======================================================================================
#
# Functions defining the evolution according to Gieles etal 2016
#
# ======================================================================================
# https://ui.adsabs.harvard.edu/abs/2016MNRAS.463L.103G/abstract
# Equation 12 is essentially the answer here.
# this model is two body relaxation plus tidal shocks. No stellar evolution.
# note their notation uses i for initial rather than 0, and I keep that in my code
print("make sure I'm doing 2-3D transition correctly")


def gieles_t_dis_equilibrium(mass):
    # t_dis is defined in equation 17. I'll assume the default value for gamma_GMC
    # NOTE THAT THIS IS ONLY ON THE EQUILIBRIUM RELATION
    return 940 * u.Myr * (mass / (10 ** 4 * u.Msun)) ** (2 / 3)


def gieles_t_sh(rho):
    # assumes default value for gamma_GMC
    gamma_gmc = 12.8 * u.Gyr
    return gamma_gmc * (rho / (100 * u.Msun / u.pc ** 3))


# This commented equation does not work, as it uses timescales only valid on the
# equilibrium relation, which is not what I have to start
# def gieles_mass_loss(initial_mass, time):
#     # all quantities should have units
#     # the last paragraph before the conclusion shows how this works.
#     # M_dot / M = 1 / t_dis
#     # M_dot = M / t_dis
#     # I'll numerically integrate this
#     dt = 3 * u.Myr
#     t_now = 0 * u.Myr
#     M_now = initial_mass.copy()
#     assert int(time / dt) == time / dt
#
#     while t_now < time:
#         # calculate the instantaneous t_di
#         t_dis = gieles_t_dis(M_now)
#         # then calculate the mass loss
#         M_dot = M_now / t_dis
#         # don't let the mass go negative
#         M_now = np.maximum(0.1 * u.Msun, M_now - M_dot * dt)
#
#         t_now += dt
#     return M_now
def gieles_etal_16_evolution(initial_radius, initial_mass, end_time):
    # Use Equation 2 to get mass
    # dM = M f dE / E
    # where Equation 4 is used for shocks:
    # dE / E = -dt / tau_sh
    # I'll numerically integrate this
    # then equation 12 to get radius at a given mass

    # use shorthands for the initial values
    r_i = initial_radius
    M_i = initial_mass
    t_end = end_time
    rho_i = calculate_density(M_i, r_i)
    rho_now = rho_i

    dt = 1 * u.Myr
    t_now = 0 * u.Myr
    M_now = initial_mass.copy()

    t_history = [t_now.copy()]
    M_history = [M_now.copy()]
    rho_history = [rho_now.copy()]
    f = 3  # default value
    while t_now < t_end:
        # calculate the shock timescale. Mass evolution only comes from shocks,
        # not two-body relaxation
        tau_sh = gieles_t_sh(rho_now)
        # we then use this to determine the mass loss
        dE_E = -dt / tau_sh
        dM = M_now * f * dE_E
        # don't let the mass go negative
        M_now = np.maximum(0.1 * u.Msun, M_now + dM)
        rho_now = gieles_etal_16_density(M_i, M_now, rho_i)

        t_now += dt

        # store variables
        t_history.append(t_now.copy())
        M_history.append(M_now.copy())
        rho_history.append(rho_now.copy())

    # turn history into nice astropy arrays
    t_history = u.Quantity(t_history)
    M_history = u.Quantity(M_history)
    rho_history = u.Quantity(rho_history)

    r_history = density_to_half_mass_radius(rho_history, M_history)
    return t_history, M_history, rho_history, r_history


def gieles_etal_16_density(initial_mass, current_mass, initial_density):
    # quantities must have units

    # shorthands
    M_i = initial_mass
    M = current_mass
    rho_i = initial_density

    # use the A value used in Figure 3, see text right before section 4.2
    A = 0.02 * u.pc ** (-9 / 2) * u.Msun ** (1 / 2)
    # f is set to 3 near the end of section 2.1, also mentioned right after equation 12
    f = 3

    # then equation 12 can simply be calculated
    numerator = A * M
    denom_term_1 = A * M_i / (rho_i ** (3 / 2))
    denom_term_2 = (M / M_i) ** (17 / 2 - 9 / (2 * f))
    denominator = 1 + (denom_term_1 - 1) * denom_term_2
    rho = (numerator / denominator) ** (2 / 3)
    return rho


def calculate_density(mass, half_mass_radius):
    # when calculating the density, take (half_mass) / (4/3 pi half_mass_radius^3)
    return 3 * 0.5 * mass / (4 * np.pi * half_mass_radius ** 3)


def density_to_half_mass_radius(density, mass):
    # then turn this back into half mass radius (remember to use half the mass
    return ((3 * 0.5 * mass) / (4 * np.pi * density)) ** (1 / 3)


# ======================================================================================
# duplicate plot from paper to validate this prescription
# ======================================================================================
# test the gieles etal 16 prescription
test_rho = 30 * u.Msun / u.pc ** 3
test_mass_initial = np.logspace(2.5, 5.0, 10) * u.Msun
test_radius_initial = density_to_half_mass_radius(test_rho, test_mass_initial)

test_run = gieles_etal_16_evolution(test_radius_initial, test_mass_initial, 300 * u.Myr)
test_t_history, test_M_history, test_rho_history, test_r_history = test_run
test_idx_30 = np.where(test_t_history == 30 * u.Myr)
test_idx_300 = np.where(test_t_history == 300 * u.Myr)

test_rho_30 = test_rho_history[test_idx_30]
test_rho_300 = test_rho_history[test_idx_300]
test_m_30 = test_M_history[test_idx_30]
test_m_300 = test_M_history[test_idx_300]

fig, ax = bpl.subplots(figsize=[7, 7])
ax.scatter(
    test_mass_initial, [test_rho.to(u.Msun / u.pc ** 3).value] * 10, label="Initial"
)
ax.scatter(test_m_30, test_rho_30, label="30 Myr")
ax.scatter(test_m_300, test_rho_300, label="300 Myr")
for idx in range(10):
    test_ms = test_M_history[:, idx]
    test_rhos = test_rho_history[:, idx]
    ax.plot(test_ms, test_rhos, c=bpl.almost_black, lw=1, zorder=0)

test_A = 0.02 * u.pc ** (-9 / 2) * u.Msun ** (1 / 2)
test_eq_m = np.logspace(-1, 6, 1000) * u.Msun
test_rho_eq = (test_A * test_eq_m) ** (2 / 3)
ax.plot(
    test_eq_m.to(u.Msun).value,
    test_rho_eq.to(u.Msun / u.pc ** 3).value,
    ls=":",
    c=bpl.almost_black,
    lw=1,
    zorder=0,
)

ax.add_labels("logM", "log rho")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_limits(1e2, 1e6, 0.1, 1e4)
ax.legend()
fig.savefig("test_gieles.png")

# ======================================================================================
#
# Run clusters through this evolution
#
# ======================================================================================
mass_initial = np.logspace(3, 5, 100) * u.Msun
reff_observed_bin1 = mass_size_relation(mass_initial, *fits["age1"])
reff_observed_bin2 = mass_size_relation(mass_initial, *fits["age2"])
reff_observed_bin3 = mass_size_relation(mass_initial, *fits["age3"])

m_g10_30myr, r_g10_30myr = gieles_etal_10_evolution(
    reff_observed_bin1, mass_initial, 30 * u.Myr
)
m_g10_300myr, r_g10_300myr = gieles_etal_10_evolution(
    reff_observed_bin1, mass_initial, 300 * u.Myr
)

t_history, M_history, rho_history, r_history = gieles_etal_16_evolution(
    reff_observed_bin1, mass_initial, 300 * u.Myr
)
idx_30 = np.where(t_history == 30 * u.Myr)[0]
idx_300 = np.where(t_history == 300 * u.Myr)[0]

# not sure why this extra index is needed
m_g16_30myr = M_history[idx_30][0]
m_g16_300myr = M_history[idx_300][0]
r_g16_30myr = r_history[idx_30][0]
r_g16_300myr = r_history[idx_300][0]

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
    log_mass = np.log10(mass.to("Msun").value)
    log_r_eff = np.log10(r_eff.to("pc").value)

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


fig, ax = bpl.subplots()
ax.plot(
    mass_initial,
    reff_observed_bin1,
    c=bpl.color_cycle[0],
    label=format_params("Age: 1-10 Myr Observed", *fits["age1"]),
)
# ax.plot(
#     mass_initial,
#     reff_observed_bin2,
#     c=colors["med"],
#     label=format_params("Age: 10-100 Myr Observed", *fits["age2"]),
# )
ax.plot(
    mass_initial,
    reff_observed_bin3,
    c=bpl.color_cycle[3],
    label=format_params("Age: 100 Myr - 1 Gyr Observed", *fits["age3"]),
)

# ax.plot(
#     m_g10_30myr,
#     r_g10_30myr,
#     c=colors["med"],
#     ls=":",
#     label=format_params(
#         "Gieles+ 2010 - 30 Myr",
#         *fit_mass_size_relation(m_g10_30myr, r_g10_30myr),
#     ),
# )

ax.plot(
    m_g10_300myr,
    r_g10_300myr,
    c=bpl.color_cycle[2],
    label=format_params(
        "Gieles+ 2010 - 300 Myr",
        *fit_mass_size_relation(m_g10_300myr, r_g10_300myr),
    ),
)

# ax.plot(
#     m_g16_30myr,
#     r_g16_30myr,
#     c=colors["med"],
#     ls="--",
#     label=format_params(
#         "Gieles+ 2016 - 30 Myr",
#         *fit_mass_size_relation(m_g16_30myr, r_g16_30myr),
#     ),
# )

ax.plot(
    m_g16_300myr,
    r_g16_300myr,
    c=bpl.color_cycle[4],
    label=format_params(
        "Gieles+ 2016 - 300 Myr",
        *fit_mass_size_relation(m_g16_300myr, r_g16_300myr),
    ),
)

mru_p.format_mass_size_plot(ax)
ax.legend(loc=3, fontsize=12)
fig.savefig(plot_name)
