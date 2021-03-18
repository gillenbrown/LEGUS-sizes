from astropy.io import fits
from astropy import wcs
from astropy import units as u
import numpy as np


def _get_image(home_dir):
    # then we have to find the image we need. This will be in the drc directory, but the
    # exact name is uncertain
    galaxy_name = home_dir.name
    # NGC 5474 has a weird naming convention for the images
    if galaxy_name == "ngc5474":
        galaxy_name += "-c"

    # we need to check for different images. We prefer F555W if it exists, but if not
    # we use F606W
    # it could be one of two instruments: ACS or UVIS.
    for band in ["f555w", "f606w"]:
        for instrument in ["acs", "uvis"]:
            image_name = f"hlsp_legus_hst_{instrument}_{galaxy_name}_{band}_v1_drc.fits"
            try:  # to load the image
                hdu_list = fits.open(home_dir / image_name)
                # if it works we found our image, so we can return
                # DRC images should have the PRIMARY extension
                return hdu_list["PRIMARY"], instrument, band
            except FileNotFoundError:
                continue  # go to next band
    else:  # no break, image not found
        # try one last thing for the mosaic for NGC 5194 and NGC 5195.
        try:
            image_name = "hlsp_legus_hst_acs_ngc5194-ngc5195-mosaic_f555w_v1_sci.fits"
            hdu_list = fits.open(home_dir / image_name)
            # if it works we found our image, so we can return
            # DRC images should have the PRIMARY extension
            return hdu_list["PRIMARY"], "acs", "f555w"
        except FileNotFoundError:
            pass  # I do this so the next error raised won't appear to come from here

        raise FileNotFoundError(
            f"No f555w or f606w image found in directory:\n{str(home_dir)}"
        )


def get_drc_image(home_dir):
    image, instrument, band = _get_image(home_dir)
    image_data = image.data
    # Multiply by the exposure time to get this in units of electrons. It stars in
    # electrons per second
    image_data *= image.header["EXPTIME"]

    return image_data, instrument, band


# Set up dictionary to use as a cache to make this faster
pixel_scale_arcsec_cache = {}


def get_pixel_scale_arcsec(home_dir):
    """ Get the pixel scale in arcseconds per pixel """
    try:
        return pixel_scale_arcsec_cache[home_dir]
    except KeyError:
        pass

    image, _, _ = _get_image(home_dir)
    image_wcs = wcs.WCS(image.header)
    pix_scale = (
        wcs.utils.proj_plane_pixel_scales(image_wcs.celestial) * image_wcs.wcs.cunit
    )

    # both of these should be the same
    assert np.isclose(pix_scale[0], pix_scale[1], rtol=1e-4, atol=0)

    r_value = pix_scale[0].to("arcsecond").value
    # add this to the cache
    pixel_scale_arcsec_cache[home_dir] = r_value
    return r_value


# https://en.wikipedia.org/wiki/Distance_modulus
def distance_modulus_distance_and_err(distance_modulus, distance_modulus_error):
    distance = 10 ** (1 + 0.2 * distance_modulus) / 1e6
    distance_err = np.log(10) * 0.2 * distance * distance_modulus_error
    return distance, distance_err


def mean_with_error(*args):
    """
    Calculate the mean of certain quantities, each with an error.

    :param args: Tuples, each of the (value, error) form
    :return: tuple with the mean and error
    """
    values = [item[0] for item in args]
    errors = [item[1] for item in args]

    weights = [1 / sigma for sigma in errors]
    mean = np.average(values, weights=weights)

    # https://stats.stackexchange.com/questions/271643/uncertainty-of-a-weighted-mean-of-uncertain-observations
    # variance calculated here (assuming all observations are independent)
    variance = len(values) * (np.sum(weights)) ** -2
    return mean, np.sqrt(variance)


# I separate out the distances Ryon used from my default values, to ensure consistency
# in the comparison plots
ryon_distances_and_errs_mpc = {
    "ngc1313-e": distance_modulus_distance_and_err(28.21, 0.02),  # Jacobs et al. 2009
    "ngc1313-w": distance_modulus_distance_and_err(28.21, 0.02),  # Jacobs et al. 2009
    "ngc628-c": distance_modulus_distance_and_err(29.98, 0.28),  # Olivares et al. 2010
    "ngc628-e": distance_modulus_distance_and_err(29.98, 0.28),  # Olivares et al. 2010
}


distances_and_errs_mpc = {
    # unless otherwise indicated, distances are from Sabbi et al 2018.
    "ic4247": (5.11, 0.4),
    # IC 559 does not have a good distance in Sabbi et al 2018, as it does not have a
    # full magnitude below the TRGB. But there are only two measured distances,
    # according to NED. The other is from Tully-Fisher and has no listed error,
    # so we use Sabbi et al. 2018
    "ic559": (10.0, 0.9),
    # For NGC 1313 we use the mean of both fields from  Sabbi et al. 2018
    "ngc1313-e": mean_with_error((4.2, 0.34), (4.4, 0.35)),
    "ngc1313-w": mean_with_error((4.2, 0.34), (4.4, 0.35)),
    # Sabbi et al. 2018 says: "A combination of distance, crowding, and metallicity made
    # the estimate of the TRGB luminosity of the NGC1433 quite uncertain." The only
    # other available estimates are Tully Fisher, and the modern ones (>1990) are
    # consistent with Sabbi et al. 2018, so I just use it for consistency.
    "ngc1433": (9.1, 1.0),
    "ngc1566": (15.6, 0.6),
    "ngc1705": (5.22, 0.38),
    "ngc3344": (8.3, 0.7),
    # Sabbi et al. 2018 says: A combination of distance, crowding, and metallicity made
    # the estimate of the TRGB luminosity of the NGC3351 quite uncertain. There are many
    # other distance indicators on NED. Sabbi et sl. 2018 is a bit lower than the
    # Cepheid distances, for example, but is still roughly consistent.
    "ngc3351": (9.3, 0.9),
    "ngc3738": (5.09, 0.40),
    "ngc4242": (5.3, 0.3),
    # For NGC 4395 we use the mean of both fields from  Sabbi et al. 2018
    "ngc4395-n": mean_with_error((4.62, 0.2), (4.41, 0.33)),
    "ngc4395-s": mean_with_error((4.62, 0.2), (4.41, 0.33)),
    "ngc4449": (4.01, 0.30),
    "ngc45": (6.8, 0.5),
    "ngc4656": (7.9, 0.7),
    # For the NGC5194/5195 mosaic, we use the mean of the the 5194 NE and SW fields,
    # as these are not crowded in Sabbi et al 2018
    "ngc5194-ngc5195-mosaic": mean_with_error((7.2, 0.6), (7.6, 0.6)),
    "ngc5238": (4.43, 0.34),
    "ngc5253": (3.32, 0.25),
    "ngc5474": (6.6, 0.5),
    "ngc5477": (6.7, 0.5),
    # For NGC 628 we use the distance from the outer field
    "ngc628-c": (8.8, 0.7),
    "ngc628-e": (8.8, 0.7),
    "ngc6503": (6.3, 0.5),
    # For NGC 7793 we take the mean of both fields from Sabbi et al. 2018
    "ngc7793-e": mean_with_error((3.75, 0.28), (3.83, 0.29)),
    "ngc7793-w": mean_with_error((3.75, 0.28), (3.83, 0.29)),
    "ugc1249": (6.4, 0.5),
    "ugc4305": (3.32, 0.25),
    "ugc4459": (3.96, 0.30),
    "ugc5139": (3.83, 0.29),
    "ugc685": (4.37, 0.34),
    "ugc695": (7.8, 0.6),
    "ugc7408": (7.0, 0.5),
    "ugca281": (5.19, 0.39),
    # M31 is needed for comparison data: Wagner-Kaiser+ (2015, MNRAS, 451, 724)
    # optical bands, as used in Krumholz
    "m31": distance_modulus_distance_and_err(24.32, 0.09),
    # artificial galaxy doesn't matter, I don't use these distances
    "artificial": (1.00, 0.1),
}


def distance(data_path, ryon=False):
    if ryon:
        return ryon_distances_and_errs_mpc[data_path.name][0] * u.Mpc
    else:
        return distances_and_errs_mpc[data_path.name][0] * u.Mpc


def distance_error(data_path, ryon=False):
    if ryon:
        return ryon_distances_and_errs_mpc[data_path.name][1] * u.Mpc
    else:
        return distances_and_errs_mpc[data_path.name][1] * u.Mpc


def pixels_to_arcsec(pixels, data_path):
    """
    Convert a size in pixels to a size in arcseconds.

    This can be either a best fit value or an error.

    :param pixels: size in pixels
    :param data_path: Location of the image - needed to get the pixel scale
    :return: Size in arcseconds
    """
    return pixels * get_pixel_scale_arcsec(data_path)


def arcsec_to_pc_with_errors(
    data_path, arcsec, arcsec_error_down, arcsec_error_up, ryon=False
):
    """

    :param data_path: Location of the image - needed to get the distance
    :param arcsec: Size in arcseconds
    :param pix_error_down: Lower error of the size in pixels
    :param pix_error_up: Upper error of the size in pixels
    :param ryon: Whether or not to use the distances quoted in Ryon et al. 2017 or
                 use the updated ones from Sabbi et al 2018
    :return: Size, lower error, upper error, all in parsecs.
    """
    radians = (arcsec * u.arcsec).to("radian").value
    parsecs = radians * distance(data_path, ryon).to("pc").value

    # then the fractional error is added in quadrature to get the resulting
    # fractional error, assuming the distance error is symmetric
    frac_err_dist = distance_error(data_path, ryon) / distance(data_path, ryon)

    frac_err_arcsec_up = arcsec_error_up / arcsec
    frac_err_arcsec_down = arcsec_error_down / arcsec

    frac_err_tot_up = np.sqrt(frac_err_dist ** 2 + frac_err_arcsec_up ** 2)
    frac_err_tot_down = np.sqrt(frac_err_dist ** 2 + frac_err_arcsec_down ** 2)

    err_pc_up = parsecs * frac_err_tot_up
    err_pc_down = parsecs * frac_err_tot_down

    return parsecs, err_pc_down, err_pc_up
