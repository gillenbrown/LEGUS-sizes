"""
format_catalogs.py

The LEGUS cluster catalogs aren't in the greatest format. The headers are separate from
the data. I want to fix that, and merge the two to create nicely formatted catalogs.
"""
from pathlib import Path
import sys

from astropy import table

# The big thing here is to use the header files to get actual column names. I'll have
# a giant function that will convert the descriptions from the header file into short
# column names
# this variable is short for `description_to_colname`
dc = {
    "source id": "ID",
    "x coordinates in the ref single frame (image aligned and registered, these coordinates are the same in each filter)": "x_pix_single",
    "y coordinates in the ref single frame (image aligned and registered, these coordinates are the same in each filter)": "y_pix_single",
    "x coordinates in the ref frame (image aligned and registered, these coordinates are the same in each filter)": "x_pix_single",
    "y coordinates in the ref frame (image aligned and registered, these coordinates are the same in each filter)": "y_pix_single",
    "x coordinates in the ref mosaic frame (image aligned and registered, these coordinates are the same in each filter)": "x_pix_mosaic",
    "y coordinates in the ref mosaic frame (image aligned and registered, these coordinates are the same in each filter)": "y_pix_mosaic",
    "RA coordinates in the ref single frame (image aligned and registered, these coordinates are the same in each filter). The RA and DEC are the same in mosaic and single frames": "RA",
    "DEC coordinates in the ref single frame (image aligned and registered, these coordinates are the same in each filter). The RA and DEC are the same in mosaic and single frames": "Dec",
    "RA coordinates in the ref frame (image aligned and registered, these coordinates are the same in each filter)": "RA",
    "DEC coordinates in the ref frame (image aligned and registered, these coordinates are the same in each filter)": "Dec",
    "final total mag in WFC3/F275W": "mag_F275W",
    "final photometric error in WFC3/F275W": "photoerr_F275W",
    "final photometric error in F275W": "photoerr_F275W",
    "final total mag in  WFC3/F336W": "mag_F336W",  # has an extra space
    "final total mag in WFC3/F336W": "mag_F336W",
    "final photometric error in WFC3/F336W": "photoerr_F336W",
    "final photometric error in F336W": "photoerr_F336W",
    "final total mag in ACS/F435W": "mag_F435W",
    "final photometric error in ACS/F435W": "photoerr_F435W",
    "final photometric error in F435W": "photoerr_F435W",
    "final total mag in ACS/F555W": "mag_F555W",
    "final total mag in WFC3/F555W": "mag_F555W",
    "final photometric error in ACS/F555W": "photoerr_F555W",
    "final photometric error in F555W": "photoerr_F555W",
    "final total mag in ACS/F814W": "mag_F814W",
    "final photometric error in ACS/F814W": "photoerr_F814W",
    "final photometric error in F814W": "photoerr_F814W",
    "CI=mag(1px)-mag(3px) measured in the F555W. This catalogue contains only sources with CI>=1.4.": "CI",
    "CI=mag(1px)-mag(3px) measured in the F555W. This catalogue contains only sources with CI>=1.3.": "CI",
    "best age in yr": "age_yr",
    "max age in yr (within 68 % confidence level)": "age_yr_max",
    "min age in yr (within 68 % confidence level)": "age_yr_min",
    "best mass in solar masses": "mass_msun",
    "max mass in solar masses (within 68 % confidence level)": "mass_msun_max",
    "min mass in solar masses (within 68 % confidence level)": "mass_msun_min",
    "best E(B-V)": "E(B-V)",
    "max E(B-V) (within 68 % confidence level)": "E(B-V)_max",
    "min E(B-V) (within 68 % confidence level)": "E(B-V)_min",
    "chi2 fit residual in F275W, if positive the flux observed at that wavelenght is higher then predicted by the best fitted model (and viceversa)": "chi_2_F265W",
    "chi2 fit residual in F336W, if positive the flux observed at that wavelenght is higher then predicted by the best fitted model (and viceversa)": "chi_2_F336W",
    "chi2 fit residual in F435W, if positive the flux observed at that wavelenght is higher then predicted by the best fitted model (and viceversa)": "chi_2_F435W",
    "chi2 fit residual in F555W, if positive the flux observed at that wavelenght is higher then predicted by the best fitted model (and viceversa)": "chi_2_F555W",
    "chi2 fit residual in F814W, if positive the flux observed at that wavelenght is higher then predicted by the best fitted model (and viceversa)": "chi_2_F814W",
    "reduced chi2": "chi_2_reduced",
    "Q probability is a measurement of the quality of the fit; if close to 1 fit is good, if close to 0 the fit outputs are not well constrained. See numerical recipies": "Q_probability",
    "Number of filter. Sources with UBVI or UV-BVI detection have Nflt=4; sources with detection in UV-UBVI have Nflt=5. The remining sources have Nflt=0. The SED fit has been done only on sources with Nflt>=4": "N_filters",
    "Final assigned class of the source after visual inspection, applying the mode. Only clusters with Nflt>=4, CI>=1.4 and m_555<=-6.0  mag have been visually inspected. Please, notice that the aperture correction applied to the V band before the magnitude cut is applied is dependent of the CI measured in this filter. Those sources which did not pass the cut have assigned a flag=0. Class=1, symmetric, compact cluster.  Class=2, concentrated object with some degree of asymmetry, possible color gradient.  Class=3, multiple peak system, diffuse, could be spurious nearby stars along the line of sight.  Class=4, spurious detection (foreground/background sources, single bright stars, artifacts).": "class_mode",
    "Final assigned class of the source after visual inspection, applying the mode. Only clusters with Nflt>=4, CI>=1.3 and m_555<=-6.0  mag have been visually inspected. Please, notice that the aperture correction applied to the V band before the magnitude cut is applied is dependent of the CI measured in this filter. Those sources which did not pass the cut have assigned a flag=0. Class=1, symmetric, compact cluster.  Class=2, concentrated object with some degree of asymmetry, possible color gradient.  Class=3, multiple peak system, diffuse, could be spurious nearby stars along the line of sight.  Class=4, spurious detection (foreground/background sources, single bright stars, artifacts).": "class_mode",
    "Final assigned class of the source after visual inspection, applying the mean. The classification is the same as before (0,1,2,3,4). A strong deviation between Class_mode and Class_mean for the same source shows uncertainty in the visual classification.": "class_mean",
}


def header_line_to_colnames(header_line):
    """
    Turn a line from the header into the old and new column names

    :param header_line: A line from the header file with info about a column. This
                        is assumed to be of the form `number. description`
    :return: A two item tuple containing the old column name and new column name
    :rtype: tuple
    """
    position = header_line.split(".")[0]
    description = ".".join(header_line.split(".")[1:]).strip()

    old_col = f"col{position}"
    new_col = dc[description]

    return old_col, new_col


def find_catalogs(home_dir):
    """
    Find the name of the base catalog name. We need a function for this because we
    don't know whether it has ACS and WFC3 in the filename of just one.

    :param home_dir: Directory to search for catalogs
    :type home_dir: Path
    :return: Path objects pointing to the catalog and readme file.
    :rtype: tuple
    """
    for item in home_dir.iterdir():
        if not item.is_file():
            continue
        filename = item.name
        # see if it starts and ends with what the catalog should be
        if filename.startswith("hlsp_legus_hst_") and filename.endswith(
            "_multiband_v1_padagb-mwext-avgapcor.tab"
        ):
            catalog = item
            header = item.with_suffix(".readme.txt")
            return catalog, header


# --------------------------------------------------------------------------------------
#
# Actual process
#
# --------------------------------------------------------------------------------------
# start by getting the catalog and header files
final_catalog = Path(sys.argv[1])
home_dir = final_catalog.parent
galaxy_name = home_dir.name
base_cat_ = f"hlsp_legus_hst_wfc3_{galaxy_name}_multiband_v1_padagb-mwext-avgapcor"
catalog_name, header_name = find_catalogs(home_dir)

# Then we can do what we need. First we'll go through the header to get the lists of
# all the old and new colnames
old_colnames = []
new_colnames = []
with open(header_name, "r") as header:
    for line in header:
        line = line.strip()

        if line.startswith("#") or len(line) == 0:
            continue
        # otherwise, see if this is a line containing info. We'll pick lines with
        # a first item that is an integer followed by a period
        first_item = line.split()[0]
        if not first_item[-1] == ".":
            continue
        try:
            int(first_item[:-1])
        except ValueError:  # unsucccessful type conversion
            continue  # go to next line

        # if we are here we have an integer followed by a period, so we have the
        # desired line
        old_col, new_col = header_line_to_colnames(line)

        old_colnames.append(old_col)
        new_colnames.append(new_col)

# Then we can use these column names to replace the ones given originally
catalog = table.Table.read(catalog_name, format="ascii")
catalog.rename_columns(old_colnames, new_colnames)
# then write this catalog to the desired output file. Astropy recommends ECSV
catalog.write(final_catalog, format="ascii.ecsv")
