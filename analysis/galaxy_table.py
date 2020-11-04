"""
galaxy_table.py - Create a table showing the characteristics of the various galaxies
and the clusters in them

This takes the following parameters
- Path where this text file will be saved
- Oversampling factor for the psf
- The width of the psf snapshot
- The source of the psf stars
- All the completed catalogs
"""
import sys
from pathlib import Path

import numpy as np
from scipy import interpolate, optimize
from astropy.io import fits
from astropy import table

# need to add the correct path to import utils
code_home_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(code_home_dir / "pipeline"))
import utils

# ======================================================================================
#
# Load basic parameters and the catalogs
#
# ======================================================================================
output_name = Path(sys.argv[1])
oversampling_factor = int(sys.argv[2])
psf_width = int(sys.argv[3])
psf_source = sys.argv[4]

catalogs = dict()
for item in sys.argv[5:]:
    cat = table.Table.read(item, format="ascii.ecsv")
    home_dir = Path(item).parent.parent
    catalogs[home_dir] = cat

big_catalog = table.vstack(list(catalogs.values()), join_type="inner")
# ======================================================================================
#
# Functions to calculate the psf effective radius
#
# ======================================================================================
def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def measure_psf_reff(home_dir):
    psf_name = (
        f"psf_"
        f"{psf_source}_stars_"
        f"{psf_width}_pixels_"
        f"{oversampling_factor}x_oversampled.fits"
    )

    psf = fits.open(home_dir / "size" / psf_name)["PRIMARY"].data
    # the center is the brightest pixel of the image
    x_cen, y_cen = np.unravel_index(np.argmax(psf), psf.shape)
    total = np.sum(psf)
    half_light = total / 2.0
    # then go through all the pixel values to determine the distance from the center.
    # Since we're using the center pixel (on an integer value), different pixels
    # will have identical distances. Add all the pixels at the same distance together
    unique_values = dict()
    for x in range(psf.shape[1]):
        for y in range(psf.shape[0]):
            # need to include the oversampling factor in the distance
            radius = distance(x, y, x_cen, y_cen) / oversampling_factor
            if radius not in unique_values:
                unique_values[radius] = 0
            unique_values[radius] += psf[y][x]

    # Then sort the radii
    radii, values = [], []
    for r, v in unique_values.items():
        radii.append(r)
        values.append(v)
    idxs_sort = np.argsort(radii)
    sorted_radii = np.array(radii)[idxs_sort]
    sorted_values = np.array(values)[idxs_sort]

    # then turn this into cumulative light
    cumulative_light = np.cumsum(sorted_values)
    # then to find where we cross 1/2, create an interpolation object
    cum_interp = interpolate.interp1d(x=sorted_radii, y=cumulative_light, kind="linear")
    # find where it reaches half
    def to_minimize(r_half):
        return abs(cum_interp(r_half) - half_light)

    psf_size_pixels = optimize.minimize(to_minimize, x0=1.5, bounds=[[0, 5]]).x[0]

    # then convert to pc
    psf_size_arcsec = utils.pixels_to_arcsec(psf_size_pixels, home_dir)
    return utils.arcsec_to_pc_with_errors(home_dir, psf_size_arcsec, 0, 0, False)[0]


# Then go ahead and calculate the effective radius
psf_sizes = {home_dir: measure_psf_reff(home_dir) for home_dir in catalogs}

# ======================================================================================
#
# Then we print this all as a nicely formatted latex table
#
# ======================================================================================
def format_galaxy_name(home_dir):
    name = home_dir.name
    # check one edge case
    if name == "ngc5194-ngc5195-mosaic":
        return "NGC 5194/NGC 5195"
    if "ngc" in name:
        return name.replace("ngc", "NGC ")
    elif "ugca" in name:
        return name.replace("ugca", "UGCA ")
    elif "ugc" in name:
        return name.replace("ugc", "UGC ")
    elif "ic" in name:
        return name.replace("ic", "IC ")
    else:
        raise ValueError(f"{name} wasn't handled properly.")


def get_iqr_string(cat):
    mask = cat["good"]
    r_eff = cat["r_eff_pc_rmax_15pix_best"][mask]
    values = np.percentile(r_eff, [25, 50, 75])
    return f"{values[0]:.2f} --- {values[1]:.2f} --- {values[2]:.2f}"


with open(output_name, "w") as out_file:
    out_file.write("\t\\begin{tabular}{lrcc}\n")
    out_file.write("\t\t\\toprule\n")
    out_file.write(
        "\t\tLEGUS Field & "
        "Number of Clusters & "
        "PSF size (pc) & "
        "Cluster $\\reff$: 25---50---75th percentiles \\\\ \n"
    )
    out_file.write("\t\t\midrule\n")
    for home_dir, cat in catalogs.items():
        galaxy = format_galaxy_name(home_dir)
        n = len(cat)
        this_psf_size = psf_sizes[home_dir]
        iqr_str = get_iqr_string(cat)

        out_file.write(f"\t\t{galaxy} & {n} & {this_psf_size:.2f} & {iqr_str} \\\\ \n")

    out_file.write("\t\t\midrule\n")
    # get the total values
    total_n = len(big_catalog)
    total_iqr = get_iqr_string(big_catalog)
    out_file.write(f"\t\tTotal & {total_n} & --- & {total_iqr} \\\\ \n")

    out_file.write("\t\t\\bottomrule\n")
    out_file.write("\t\end{tabular}\n")
