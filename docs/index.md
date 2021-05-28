# LEGUS-sizes
This repository holds the code and the catalog used in Brown & Gnedin 2021. The catalog is the `cluster_sizes_brown_gnedin_21.txt` file here. I'll first describe the catalog, then describe the code used to generate the catalog.

## Cluster Catalog
The cluster catalog contains quantities calculated by LEGUS such as masses and radii ([Calzetti et al. 2015](https://ui.adsabs.harvard.edu/abs/2015AJ....149...51C/abstract), [Adamo et al. 2017](https://ui.adsabs.harvard.edu/abs/2017ApJ...841..131A/abstract), [Cook et al. 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.484.4897C/abstract)) as well as the radii and other derived quantities from this paper. If using quantities from those papers, please cite them appropriately.

I will not give a detailed description of the columns that come from LEGUS. Reference the papers above or the [public LEGUS catalogs](https://archive.stsci.edu/prepds/legus/dataproducts-public.html) for more on that data.

I do include all properties needed to duplicate the analysis done in the paper.
**`field`**
The identifier for the LEGUS field. Note that some galaxies are split over multiple fields (NGC 1313, NGC 4395, NGC 628, and NGC 7793), and that one field contains multiple galaxies (NGC 5194 and NGC 5195).

**`ID`**
The cluster ID assigned by LEGUS. This was done on a field-by-field basis.

**`galaxy`**
The galaxy the cluster belongs to. NGC 5194 and NGC 5195 are separatedmanually (see Figure 1 of the paper). 

**`galaxy_distance_mpc`, `galaxy_distance_mpc_err`**
Distance to the galaxy and its error. We use the TRGB distances to all LEGUS galaxies provided by [Sabbi et al. 2018](https://ui.adsabs.harvard.edu/abs/2018ApJS..235...23S/abstract), except for NGC 1566. See the end of Section 2.4 for more on distances used.

**`galaxy_stellar_mass`, `galaxy_sfr`, `galaxy_ssfr`**
Stellar masses, star formation rates, and  specific star formation rates of the host galaxy, from [Calzetti et al. 2015](https://ui.adsabs.harvard.edu/abs/2015AJ....149...51C/abstract).SFR is obtained from *GALEX* far-UV corrected for dust attenuation, as described in [Lee et al. 2009](https://ui.adsabs.harvard.edu/abs/2009ApJ...706..599L/abstract), and stellar mass from extinction-corrected B-band luminosity and color information, as described in [Bothwell et al. 2009](https://ui.adsabs.harvard.edu/abs/2009MNRAS.400..154B/abstract)and using the mass-to-light ratio models of [Bell & de Jong 2001](https://ui.adsabs.harvard.edu/abs/2001ApJ...550..212B/abstract).

**`pixel_scale`**
Pixel scale for the image. All are nearly 39.62 mas/pixel.

**`RA`, `Dec`**
Right ascension and declination from the LEGUS catalog.

**`x_pix_single`, `y_pix_single`**
X/Y pixel position of the cluster from the LEGUS catalog.

**`mag_F275W`, `photoerr_F275W`, `mag_F336W`, `photoerr_F336W`, `mag_F814W`, `photoerr_F814W`**
LEGUS photometry.

**`CI`**
Concentration Index (CI), defined as the which is the magnitude difference between apertures of radius 1 pixel and 3 pixels. A cut was used to separate stars from clusters.

