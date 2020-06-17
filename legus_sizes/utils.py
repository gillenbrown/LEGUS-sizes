from astropy.io import fits
from astropy import wcs
from astropy import units as u
import numpy as np


def _get_image(home_dir):
    # then we have to find the image we need. This will be in the drc directory, but the
    # exact name is uncertain
    galaxy_name = home_dir.name
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
                return hdu_list["PRIMARY"], instrument
            except FileNotFoundError:
                continue  # go to next band
    else:  # no break, image not found
        raise FileNotFoundError(
            f"No f555w or f606w image found in directory:\n{str(home_dir)}"
        )


def get_drc_image(home_dir):
    image, instrument = _get_image(home_dir)
    image_data = image.data
    # Multiply by the exposure time to get this in units of electrons. It stars in
    # electrons per second
    image_data *= image.header["EXPTIME"]

    return image_data, instrument


# Set up dictionary to use as a cache to make this faster
pixel_scale_arcsec_cache = {}


def get_pixel_scale_arcsec(home_dir):
    """ Get the pixel scale in arcseconds per pixel """
    try:
        return pixel_scale_arcsec_cache[home_dir]
    except KeyError:
        pass

    image, _ = _get_image(home_dir)
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


def get_f555w_pixel_scale_pc(home_dir):
    arcsec_per_pixel = get_pixel_scale_arcsec(home_dir)
    return arcsec_to_size_pc(arcsec_per_pixel, home_dir)


# https://en.wikipedia.org/wiki/Distance_modulus
def distance_modulus_to_distance_mpc(distance_modulus):
    return 10 ** (1 + 0.2 * distance_modulus) / 1e6


def distance_modulus_err_to_distance_err_mpc(distance_modulus, distance_modulus_error):
    distance = distance_modulus_to_distance_mpc(distance_modulus)
    return np.log(10) * 0.2 * distance * distance_modulus_error


# http://edd.ifa.hawaii.edu/get_results_pgc.php?pgc=12286
def distance(data_path):
    distances_mpc = {
        "ngc1313-e": distance_modulus_to_distance_mpc(28.21),  # Jacobs et al. 2009
        "ngc1313-w": distance_modulus_to_distance_mpc(28.21),  # Jacobs et al. 2009
        "ngc628-c": distance_modulus_to_distance_mpc(29.98),  # Olivares et al. 2010
        "ngc628-e": distance_modulus_to_distance_mpc(29.98),  # Olivares et al. 2010
    }
    return distances_mpc[data_path.name] * u.Mpc


def distance_error(data_path):
    distances_error_mpc = {
        "ngc1313-e": distance_modulus_err_to_distance_err_mpc(28.21, 0.02),
        "ngc1313-w": distance_modulus_err_to_distance_err_mpc(28.21, 0.02),
        "ngc628-c": distance_modulus_err_to_distance_err_mpc(29.98, 0.28),
        "ngc628-e": distance_modulus_err_to_distance_err_mpc(29.98, 0.28),
    }
    return distances_error_mpc[data_path.name] * u.Mpc


def arcsec_to_size_pc(arcseconds, home_dir):
    # We need the home directory to get the distance
    radians = (arcseconds * u.arcsec).to("radian").value
    size = radians * distance(home_dir)
    return size.to("pc").value


def pixels_to_pc_with_errors(
    data_path, pix, pix_error_up, pix_error_down, include_distance_err=True
):
    arcsec_per_pixel = get_f555w_pixel_scale_arcsec(data_path)
    # this has no error
    radians_per_pixel = (arcsec_per_pixel * u.arcsec).to("radian").value

    parsecs = radians_per_pixel * pix * distance(data_path)
    parsecs = parsecs.to("pc").value

    # the fractional error is then added in quadrature. We assume the distance error
    # is symmetric
    if include_distance_err:
        frac_err_dist = distance_error(data_path) / distance(data_path)
    else:
        frac_err_dist = 0

    frac_err_pix_up = pix_error_up / pix
    frac_err_pix_down = pix_error_down / pix

    frac_err_tot_up = np.sqrt(frac_err_dist ** 2 + frac_err_pix_up ** 2)
    frac_err_tot_down = np.sqrt(frac_err_dist ** 2 + frac_err_pix_down ** 2)

    err_pc_up = parsecs * frac_err_tot_up
    err_pc_down = parsecs * frac_err_tot_down

    return parsecs, err_pc_down, err_pc_up
