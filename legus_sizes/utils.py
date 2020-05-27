from astropy.io import fits
from astropy import wcs
from astropy import units as u
import numpy as np


def _get_f555w_image(home_dir):
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
    return hdu_list["PRIMARY"], instrument


def get_f555w_drc_image(home_dir):
    image, instrument = _get_f555w_image(home_dir)
    image_data = image.data
    # Multiply by the exposure time to get this in units of electrons. It stars in
    # electrons per second
    image_data *= image.header["EXPTIME"]

    return image_data, instrument


def get_f555w_pixel_scale_arcsec(home_dir):
    """ Get the pixel scale in arcseconds per pixel """
    image, _ = _get_f555w_image(home_dir)
    image_wcs = wcs.WCS(image.header)
    pix_scale = (
        wcs.utils.proj_plane_pixel_scales(image_wcs.celestial) * image_wcs.wcs.cunit
    )

    # both of these should be the same
    assert np.isclose(pix_scale[0], pix_scale[1], rtol=1e-4, atol=0)
    return pix_scale[0].to("arcsecond").value


def get_f555w_pixel_scale_pc(home_dir):
    arcsec_per_pixel = get_f555w_pixel_scale_arcsec(home_dir)
    return arcsec_to_size_pc(arcsec_per_pixel, home_dir)


def distance(data_path):
    distances_mpc = {
        "ngc1313-e": 4.39,
        "ngc1313-w": 4.39,
        "ngc628-c": 9.9,
        "ngc628-e": 9.9,
    }
    return distances_mpc[data_path.name] * u.Mpc


def arcsec_to_size_pc(arcseconds, home_dir):
    # We need the home directory to get the distance
    radians = (arcseconds * u.arcsec).to("radian").value
    size = radians * distance(home_dir)
    return size.to("pc").value
