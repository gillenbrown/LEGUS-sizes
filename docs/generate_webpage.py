"""
generate_webpage.py

Generate the readme file. I only do this programatically to make the formatting easier
on me and to make sure I document all columns in the public catalog.

Takes the following command line arguments:
- location to save the readme
- the public catalog.
"""
import sys
import re
from pathlib import Path
from astropy import table

# Get the input arguments
out_file = open(sys.argv[1], "w")
catalog = table.Table.read(sys.argv[2], format="ascii.ecsv")
template_loc = sys.argv[3]
example_script_loc = sys.argv[4]

# get the name of the catalog
public_cat_name = Path(sys.argv[2]).resolve().name

# set up the column names to make sure I include all of them, and that I do so in order
unused_colnames = catalog.colnames.copy()

# ======================================================================================
#
# read in the example script ahead of time so it can be inserted later
#
# ======================================================================================
example_script_text = ""
with open(example_script_loc, "r") as example:
    # I want to ignore some lines, which I've noted in the script with tags
    ignore = False
    for line in example:
        # check for the flags
        if line.strip() == "# ignore":
            ignore = True
        elif line.strip() == "# end ignore":
            ignore = False
            continue  # don't write this flag line

        # then we can use this to hold the lines
        if not ignore:
            example_script_text += line

# I also have a placeholder for the name of the catalog, which needs to be replaced
example_script_text = example_script_text.replace(
    "cat_loc_replace", f'"{public_cat_name}"'
)

# ======================================================================================
#
# set up the items to group together
#
# ======================================================================================
groups = {
    "field": ["field"],
    "id": ["ID"],
    "galaxy": ["galaxy"],
    "distance": ["galaxy_distance_mpc", "galaxy_distance_mpc_err"],
    "galaxy_props": ["galaxy_stellar_mass", "galaxy_sfr", "galaxy_ssfr"],
    # "pixel_scale": ["pixel_scale"],
    "ra_dec": ["RA", "Dec"],
    "xy_legus": ["x_pix_single", "y_pix_single"],
    "morphology": ["morphology_class"],
    "morphology_source": ["morphology_class_source"],
    "age": ["age_yr", "age_yr_min", "age_yr_max"],
    "mass": ["mass_msun", "mass_msun_min", "mass_msun_max"],
    "xy": [
        "x_fitted",
        "x_fitted_e-",
        "x_fitted_e+",
        "y_fitted",
        "y_fitted_e-",
        "y_fitted_e+",
    ],
    "mu": ["mu_0", "mu_0_e-", "mu_0_e+"],
    "a": ["scale_radius_pixels", "scale_radius_pixels_e-", "scale_radius_pixels_e+"],
    "q": ["axis_ratio", "axis_ratio_e-", "axis_ratio_e+"],
    "theta": ["position_angle", "position_angle_e-", "position_angle_e+"],
    "eta": ["power_law_slope", "power_law_slope_e-", "power_law_slope_e+"],
    "bg": ["local_background", "local_background_e-", "local_background_e+"],
    "bootstrap": ["num_bootstrap_iterations"],
    "fit_failure": ["radius_fit_failure"],
    "prof_diff": ["profile_diff_reff"],
    "reliable_radius": ["reliable_radius"],
    "reliable_mass": ["reliable_mass"],
    "r_eff_pix": ["r_eff_pixels", "r_eff_pixels_e-", "r_eff_pixels_e+"],
    "r_eff_arcsec": ["r_eff_arcsec", "r_eff_arcsec_e-", "r_eff_arcsec_e+"],
    "r_eff_pc": ["r_eff_pc", "r_eff_pc_e-", "r_eff_pc_e+"],
    "crossing_time": ["crossing_time_yr", "crossing_time_log_err"],
    "density": ["density", "density_log_err"],
    "surface_density": ["surface_density", "surface_density_log_err"],
}

# ======================================================================================
#
# set up the descriptions for each group
#
# ======================================================================================
descriptions = {
    "field": "The identifier for the LEGUS field. Note that some galaxies are split "
    "over multiple fields (NGC 1313, NGC 4395, NGC 628, and NGC 7793), and that one "
    "field contains multiple galaxies (NGC 5194 and NGC 5195).",
    "id": "The cluster ID assigned by LEGUS. This was done on a field-by-field basis.",
    "galaxy": "The galaxy the cluster belongs to. NGC 5194 and NGC 5195 are separated "
    "manually (see Figure 1 of the paper). ",
    "distance": "Distance to the galaxy and its error. We use the TRGB distances to "
    "all LEGUS galaxies provided by "
    "[Sabbi et al. 2018]"
    "(https://ui.adsabs.harvard.edu/abs/2018ApJS..235...23S/abstract)"
    ", except for NGC 1566. See the end of Section 2.4 for more on distances used.",
    "galaxy_props": "Stellar masses, star formation rates, and  specific star "
    "formation rates of the host galaxy, from "
    "[Calzetti et al. 2015]"
    "(https://ui.adsabs.harvard.edu/abs/2015AJ....149...51C/abstract). "
    "SFR is obtained from *GALEX* far-UV corrected for dust attenuation, as described "
    "in "
    "[Lee et al. 2009](https://ui.adsabs.harvard.edu/abs/2009ApJ...706..599L/abstract)"
    ", and stellar mass from extinction-corrected B-band luminosity and color "
    "information, as described in "
    "[Bothwell et al. 2009]"
    "(https://ui.adsabs.harvard.edu/abs/2009MNRAS.400..154B/abstract) "
    "and using the mass-to-light ratio models of "
    "[Bell & de Jong 2001]"
    "(https://ui.adsabs.harvard.edu/abs/2001ApJ...550..212B/abstract). ",
    # "pixel_scale": "Pixel scale for the image. All are nearly 39.62 mas/pixel.",
    "ra_dec": "Right ascension and declination from the LEGUS catalog.",
    "xy_legus": "X/Y pixel position of the cluster from the LEGUS catalog.",
    # "ci": "Concentration Index (CI), defined as the which is the magnitude difference "
    # "between apertures of radius 1 pixel and 3 pixels. A cut was used to separate "
    # "stars from clusters.",
    "morphology": "Visual classification of the morphology of the clusters by LEGUS "
    "team members. Three or more team members visually inspect each cluster candidate, "
    "classifying it into one of the following four classes. Class 1 objects are "
    "compact and centrally concentrated with a homogeneous color. Class 2 clusters "
    "have slightly elongated density profiles and a less symmetric light distribution. "
    "Class 3 clusters are likely compact associations, having asymmetric profiles or "
    "multiple peaks on top of diffuse underlying wings. Class 4 objects are stars or "
    "artifacts. Note that a handful of galaxies have classifications from machine "
    "learning. See the `morphology_class_source` attribute to see which galaxies this "
    "applies to.",
    "morphology_source": "The source of the classification of the morphology in the "
    "`morphology` attribute. When available, we use the mode of the classifications "
    "from multiple team members, called `human_mode` in this column. Additionally, "
    "machine learning classifications (`ml`) are available for several galaxies "
    "For NGC 5194 and NGC 5195, we use the human classifications for clusters where "
    "those are available, and supplement with machine learning classifications for "
    "clusters not inspected by humans. In NGC 1566, we use the hybrid classification "
    "system (`hybrid`) created by the LEGUS team, where some clusters are inspected "
    "by humans only, some by machine learning only, and some with a machine learning "
    "classification verified by humans.",
    "age": "Cluster age (in years) and its minimum and maximum allowed value from "
    "LEGUS. This uses the deterministic SED fitting method presented in "
    "[Adamo et al. 2017]"
    "(https://ui.adsabs.harvard.edu/abs/2017ApJ...841..131A/abstract).",
    "mass": "Cluster mass (in solar masses) and its minimum and maximum allowed value "
    "from LEGUS using the same SED fitting as `age`. ",
    "xy": "The x/y pixel position. ",
    "mu": "The central pixel value $\mu_0$, in units of electrons. Note that this is "
    "the peak pixel value of the raw profile before convolution with the PSF and "
    "rebinning (see Equation 8), so it may not be directly useful.",
    "a": "Scale radius $a$, in units of pixels.",
    "q": "Axis ratio $q$, defined as the ratio of the minor to major axis, such that "
    "$0 < q \leq 1$.",
    "theta": "Position angle $\\theta$.",
    "eta": "Power law slope $\eta$.",
    "bg": "Value of the local background, in units of electrons.",
    "bootstrap": "Number of bootstrap iterations done to calculate errors on "
    "fit parameters.",
    "fit_failure": "Whether a given cluster is identified as having a failed radius "
    "fit. We define this as as a scale radius $a < 0.1$ pixels, $a > 15$ pixels, or an "
    "axis ratio $q < 0.3$. We also exclude any clusters where the fitted center is "
    "more than 2 pixels away from the central pixel identified by LEGUS.",
    "prof_diff": "Our metric to evaluate the fit quality, defined in Equation 16. "
    "It uses the cumulative light profile to estimate the half-light radius of the "
    "cluster non-parametrically, then compares the enclosed light of the model and "
    "data within this radius. This value is the fractional error of the enclosed "
    "light of the model. We use this quantity to determine whether the radius fit "
    "is reliable (Section 2.6).",
    "reliable_radius": "Whether or not this cluster radius is deemed to be reliable. "
    "To be reliable, a cluster must not have a failed fit (see above), and must not "
    "be in the worst 10th percentile of `prof_diff`. See Section 2.6 for more on this. "
    "Our analysis in the paper only uses clusters deemed to be reliable.",
    "reliable_mass": "Whether or not we consider this cluster to have a reliable "
    "measurement of the mass. This relies on a consideration of the Q statistic "
    "(see Section 3.3 for more on this). For any analysis using masses or ages, we "
    "only consider clusters with reliable masses.",
    "r_eff_pix": "The cluster effective radius, or more precisely the projected "
    "half light radius, in units of pixels. See Section 2.5 for more on how this is "
    "calculated.",
    "r_eff_arcsec": "The cluster effective radius, or more precisely the projected "
    "half light radius, in units of arcseconds. We provide this to make it easier for "
    "future users (i.e. you) to modify the galaxy distance estimates assumed in this "
    "paper.",
    "r_eff_pc": "The cluster effective radius, or more precisely the projected "
    "half light radius, in units of parsecs. The galaxy distances in this table were "
    "used to convert from arcseconds to parsecs.",
    "crossing_time": "The cluster crossing time, as defined by "
    "[Gieles & Portegies Zwart 2011]"
    "(https://ui.adsabs.harvard.edu/abs/2011MNRAS.410L...6G/abstract) "
    "(our equation 21). ",
    "density": "The cluster average 3D mass density within the half light radius, as "
    "defined by Equation 22, in units of $M_\odot pc^{-3}$",
    "surface_density": "The cluster average surface mass density within the half "
    "light radius, as defined by Equation 22, in units of $M_\odot pc^{-2}$",
}

assert sorted(list(groups.keys())) == sorted(list(descriptions.keys()))

# ======================================================================================
#
# write these to the catalog!
#
# ======================================================================================
# What I'll do is have placeholders for specific column. I'll identify those
# placeholders and replace them with the actual column name
template_flag = "__template__"
template_regex = re.compile(f"{template_flag}\w*{template_flag}")

with open(template_loc, "r") as template:
    for line in template:
        # see if this is a line where a template should be inserted
        if template_regex.match(line.strip()):
            # grab the name of the group being indicated here
            group_name = line.strip().replace(template_flag, "")
            # then get the columns that belong to this group
            group = groups[group_name]

            # validate that we got the colnames in order as we write them
            for col in group:
                try:
                    assert col == unused_colnames[0]
                except AssertionError:
                    raise RuntimeError(f"{col} bad order, should be:", unused_colnames)
                # if successful, delete the column
                del unused_colnames[0]

            # then go through and write the column names to the file.
            out_file.write("**")
            for idx_c in range(len(group)):
                out_file.write(f"`{group[idx_c]}`")
                # if we are not at the last column name of the group, we need to
                # write a comma
                if idx_c != len(group) - 1:
                    out_file.write(", ")
                # if we are the last column name, put the newline
                else:
                    out_file.write("**\n\n")

            # then write the description
            out_file.write(descriptions[group_name])
            out_file.write("\n\n")
        # I also have the code example
        elif line.strip() == "__insert_code_here__":
            out_file.write(example_script_text)

        # if it does not match any templates, just write the line
        else:
            out_file.write(line)

# check that we got everything
try:
    assert len(unused_colnames) == 0
except AssertionError:
    raise RuntimeError("These coluns are unused:", unused_colnames)

out_file.close()
