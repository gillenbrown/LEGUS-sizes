from astropy.io import fits
from astropy import stats


def get_f555w_drc_image(home_dir):
    # then we have to find the image we need. This will be in the drc directory, but the
    # exact name is uncertain
    galaxy_name = home_dir.name
    image_dir = home_dir / f"{galaxy_name}_drc"
    # it could be one of two instruments: ACS or UVIS.
    for instrument in ["acs", "uvis"]:
        image_name = f"hlsp_legus_hst_{instrument}_{galaxy_name}_f555w_v1_drc.fits"
        try:  # to load the image
            hdu_list = fits.open(image_dir / image_name)
            # if it works break out of this
            break
        except FileNotFoundError:
            continue  # go to next band
    else:  # no break, image not found
        raise FileNotFoundError(f"No f555w image found in directory:\n{str(image_dir)}")

    # DRC images should have the PRIMARY extension
    image_data = hdu_list["PRIMARY"].data

    # Calculate the noise level, which will be used to adjust plots. Also subtract off the
    # background
    _, median, _ = stats.sigma_clipped_stats(image_data, sigma=2.0)
    image_data -= median

    return image_data, instrument
