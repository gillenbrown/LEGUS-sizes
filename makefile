# Figure out what machine we're on
hostname = $(shell hostname)
# Then match this to my machines.
# findstring returns the matching part of the string. If it's not empty when
# we try to find the shangrila hostname, we know we're on shangrila
ifneq (,$(findstring shangrila,$(hostname)))
    data_home = /u/home/gillenb/code/mine/LEGUS-sizes/data
endif
ifneq (,$(findstring gillenb-mbp,$(hostname)))
    data_home = /Users/gillenb/code/legus_sizes/data
endif
# This directory should have nothing but directories with data
# We'll do this complicated line that just gets all directories inside data_home
all_data_dirs = $(sort $(dir $(wildcard $(data_home)/*/)))
data_dirs = $(filter-out %artificial/, $(all_data_dirs))
ryon_dirs = $(filter %ngc1313-e/ %ngc1313-w/ %ngc628-e/ %ngc628-c/, $(all_data_dirs))
artificial_dir = $(filter %artificial/, $(all_data_dirs))

# ------------------------------------------------------------------------------
#
# Configuration variables
#
# ------------------------------------------------------------------------------
# psf type can either be "my" or "legus"
psf_type = my
psf_pixel_size = 15
psf_oversampling_factor = 2
fit_region_size = 30
run_name = final

# ------------------------------------------------------------------------------
#
# Python scripts
#
# ------------------------------------------------------------------------------
pipeline_dir = ./pipeline/
analysis_dir = ./analysis/
mass_radius_dir = $(analysis_dir)mass_radius_relation/

catalog_script = $(pipeline_dir)format_catalogs.py
v1_star_list_script = $(pipeline_dir)preliminary_star_list.py
psf_star_list_script = $(pipeline_dir)select_psf_stars.py
psf_creation_script = $(pipeline_dir)make_psf.py
psf_comparison_script = $(analysis_dir)psf_compare.py
psf_demo_image_script = $(analysis_dir)psf_demo_image.py
sigma_script = $(pipeline_dir)make_sigma_image.py
mask_script = $(pipeline_dir)make_mask_image.py
fitting_script = $(pipeline_dir)fit.py
fit_utils = $(pipeline_dir)fit_utils.py
final_catalog_script = $(pipeline_dir)derived_properties.py
comparison_script = $(analysis_dir)ryon_comparison.py
radii_def_plot_script = $(analysis_dir)radii_def_comp_plot.py
parameters_dist_script = $(analysis_dir)parameter_distribution.py
all_galaxies_script = $(analysis_dir)all_galaxies_hist.py
all_galaxies_iso_script = $(analysis_dir)all_galaxies_hist_isolate.py
stacked_distribution_script = $(analysis_dir)stacked_distribution.py
example_plot_script = $(analysis_dir)example_fit.py
dynamical_age_script = $(analysis_dir)dynamical_age.py
density_script = $(analysis_dir)density.py
toy_model_script = $(analysis_dir)age_toy_model.py
fit_quality_script = $(analysis_dir)fit_quality.py
galaxy_table_script = $(analysis_dir)galaxy_table.py
experiment_script = ./testing/experiments.py
mass_radius_utils = $(mass_radius_dir)mass_radius_utils.py
mass_radius_utils_mle_fitting = $(mass_radius_dir)mass_radius_utils_mle_fitting.py
mass_radius_utils_plotting = $(mass_radius_dir)mass_radius_utils_plotting.py
mass_radius_utils_external_data = $(mass_radius_dir)mass_radius_utils_external_data.py
mass_radius_legus_full_script = $(mass_radius_dir)mass_radius_legus_full.py
mass_radius_legus_young_script = $(mass_radius_dir)mass_radius_legus_young.py
mass_radius_legus_agesplit_script = $(mass_radius_dir)mass_radius_legus_agesplit.py
mass_radius_legus_ssfrsplit_script = $(mass_radius_dir)mass_radius_legus_ssfrsplit.py
mass_radius_legus_mw_script = $(mass_radius_dir)mass_radius_legus_mw.py
mass_radius_legus_external_script = $(mass_radius_dir)mass_radius_legus_external.py
mass_radius_legus_mw_external_script = $(mass_radius_dir)mass_radius_legus_mw_external.py
mass_radius_final_table_script = $(mass_radius_dir)mass_radius_final_table.py
artificial_cluster_catalog_script = $(pipeline_dir)artificial_cluster_catalog.py
artificial_cluster_image_script = $(pipeline_dir)artificial_cluster_image.py
artificial_comparison_script = $(analysis_dir)artificial_comparison.py

# ------------------------------------------------------------------------------
#
# Directories to store data
#
# ------------------------------------------------------------------------------
my_dirname = size/
my_dirs = $(foreach dir,$(data_dirs),$(dir)$(my_dirname))
my_dirs_ryon = $(foreach dir,$(ryon_dirs),$(dir)$(my_dirname))
cluster_fit_dirs = $(foreach dir,$(my_dirs),$(dir)cluster_fit_plots)
cluster_plot_dirs = $(foreach dir,$(my_dirs),$(dir)plots)
local_plots_dir = ./outputs_$(run_name)/
mass_size_tables_dir = $(local_plots_dir)sub_fit_tables/
all_my_dirs = $(my_dirs) $(cluster_fit_dirs) $(cluster_plot_dirs) $(local_plots_dir) $(mass_size_tables_dir)

# ------------------------------------------------------------------------------
#
# All the pipeline filenames I'll produce
#
# ------------------------------------------------------------------------------
cat = clean_catalog.txt
star_prelim = preliminary_stars.txt
star_psf = psf_stars.txt
psf_legus = psf_legus_stars_$(psf_pixel_size)_pixels_$(psf_oversampling_factor)x_oversampled.fits
psf_my = psf_my_stars_$(psf_pixel_size)_pixels_$(psf_oversampling_factor)x_oversampled.fits
psf_comp_plot = psf_paper_my_stars_$(psf_pixel_size)_pixels_$(psf_oversampling_factor)x_oversampled.png
sigma_image = sigma_electrons.fits
mask = mask_image.fits
fit = cluster_fits_$(run_name)_$(fit_region_size)_pixels_psf_$(psf_type)_stars_$(psf_pixel_size)_pixels_$(psf_oversampling_factor)x_oversampled.h5
final_cat = final_catalog_$(run_name)_$(fit_region_size)_pixels_psf_$(psf_type)_stars_$(psf_pixel_size)_pixels_$(psf_oversampling_factor)x_oversampled.txt
fit_ryon = cluster_fits_$(run_name)_$(fit_region_size)_pixels_psf_$(psf_type)_stars_$(psf_pixel_size)_pixels_$(psf_oversampling_factor)x_oversampled_ryonlike.h5
final_cat_ryon = final_catalog_$(run_name)_$(fit_region_size)_pixels_psf_$(psf_type)_stars_$(psf_pixel_size)_pixels_$(psf_oversampling_factor)x_oversampled_ryonlike.txt

# ------------------------------------------------------------------------------
#
# Put all these files in each of these directories
#
# ------------------------------------------------------------------------------
catalogs = $(foreach dir,$(my_dirs),$(dir)$(cat))
v1_star_lists = $(foreach dir,$(my_dirs),$(dir)$(star_prelim))
psf_star_lists = $(foreach dir,$(my_dirs),$(dir)$(star_psf))
psfs_legus = $(foreach dir,$(my_dirs),$(dir)$(psf_legus))
psfs_my = $(foreach dir,$(my_dirs),$(dir)$(psf_my))
psf_comp_plots = $(foreach dir,$(my_dirs),$(dir)$(psf_comp_plot))
sigma_images = $(foreach dir,$(my_dirs),$(dir)$(sigma_image))
masks = $(foreach dir,$(my_dirs),$(dir)$(mask))
fits = $(foreach dir,$(my_dirs),$(dir)$(fit))
final_cats = $(foreach dir,$(my_dirs),$(dir)$(final_cat))
fits_ryon = $(foreach dir,$(my_dirs_ryon),$(dir)$(fit_ryon))
final_cats_ryon = $(foreach dir,$(my_dirs_ryon),$(dir)$(final_cat_ryon))

# determine which psfs to use for fitting
ifeq ($(psf_type),my)
fit_psf = $(psf_my)
fit_psfs = $(psfs_my)
else ifeq ($(psf_type),legus)
fit_psf = $(psf_legus)
fit_psfs = $(psfs_legus)
else
$(error Bad PSF type!)
endif

# ------------------------------------------------------------------------------
#
# Various plots and tables that will be here in this directory
#
# ------------------------------------------------------------------------------
galaxy_table = $(local_plots_dir)galaxy_table.txt
psf_demo_image = $(local_plots_dir)psf_demo_$(psf_type)_stars_$(psf_pixel_size)_pixels_$(psf_oversampling_factor)x_oversampled.pdf
comparison_plot = $(local_plots_dir)comparison_plot.pdf
param_dist_plot = $(local_plots_dir)parameter_distribution_size.pdf
all_galaxies_plot = $(local_plots_dir)all_galaxies.pdf
all_galaxies_iso_plot = $(local_plots_dir)all_galaxies_isolate.pdf
stacked_distribution_plot = $(local_plots_dir)stacked_distribution.pdf
dynamical_age_plot = $(local_plots_dir)dynamical_age.pdf
bound_fraction_plot = $(local_plots_dir)bound_fraction.pdf
density_plot = $(local_plots_dir)density.pdf
density_fits_txt = $(local_plots_dir)density_fits.txt
toy_model_plot = $(local_plots_dir)age_toy_model.pdf
example_fit_plot = $(local_plots_dir)example_fit.pdf
fit_quality_plot = $(local_plots_dir)fit_quality.pdf
# lots of mass size versions, all done separately
mass_radius_legus_full_plot = $(local_plots_dir)mass_radius_legus_full.pdf
mass_radius_legus_full_txt = $(mass_size_tables_dir)legus_full_table.txt
mass_radius_legus_young_plot = $(local_plots_dir)mass_radius_legus_young.pdf
mass_radius_legus_young_txt = $(mass_size_tables_dir)legus_young_table.txt
mass_radius_legus_agesplit_plot = $(local_plots_dir)mass_radius_legus_agesplit.pdf
mass_radius_legus_agesplit_txt = $(mass_size_tables_dir)legus_agesplit_table.txt
mass_radius_legus_ssfrsplit_plot = $(local_plots_dir)mass_radius_legus_ssfrsplit.pdf
mass_radius_legus_ssfrsplit_txt = $(mass_size_tables_dir)legus_ssfrsplit_table.txt
mass_radius_legus_mw_txt = $(mass_size_tables_dir)legus_mw_table.txt
mass_radius_legus_external_txt = $(mass_size_tables_dir)legus_external_table.txt
mass_radius_legus_mw_external_plot = $(local_plots_dir)mass_radius_legus_mw_external.pdf
mass_radius_legus_mw_external_txt = $(mass_size_tables_dir)legus_mw_external_table.txt
# the mass size tables get combined together into one final table
mass_radius_table = $(local_plots_dir)mass_radius_fits_table.txt
# Also do a comparison of the artificial star tests
artificial_comparison = $(local_plots_dir)artificial_tests.pdf
# then combine everything together
outputs = $(galaxy_table) $(psf_demo_image) $(psf_comp_plots) \
          $(comparison_plot) $(param_dist_plot) \
          $(all_galaxies_plot) $(all_galaxies_iso_plot) $(stacked_distribution_plot) \
          $(dynamical_age_plot) $(bound_fraction_plot) \
          $(density_plot) $(density_fits_txt) \
          $(fit_quality_plot) $(toy_model_plot) $(example_fit_plot) \
          $(mass_radius_legus_full_plot) $(mass_radius_legus_full_txt) \
          $(mass_radius_legus_young_plot) $(mass_radius_legus_young_txt) \
          $(mass_radius_legus_agesplit_plot) $(mass_radius_legus_agesplit_txt) \
          $(mass_radius_legus_ssfrsplit_plot) $(mass_radius_legus_ssfrsplit_txt) \
          $(mass_radius_legus_mw_txt) \
          $(mass_radius_legus_external_txt) \
          $(mass_radius_legus_mw_external_plot) $(mass_radius_legus_mw_external_txt) \
          $(mass_radius_table) \
          $(artificial_comparison)


# ------------------------------------------------------------------------------
#
#  Rules
#
# ------------------------------------------------------------------------------
# https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html

all: $(all_my_dirs) $(outputs)

# When we clean we will only clean the things after the fitting, since that
# takes so long. The "or true" thing there stops make from throwing an error
# if the files we try to remove don't exist
.PHONY: clean
clean:
	rm $(final_cats) $(plots) || true

# have another option to remove the fits too, just not the PSFs. Be careful
# that we only remove the plots for this image size.
debug_plots = $(foreach dir,$(cluster_fit_dirs),$(dir)/*size_$(fit_region_size)*)
.PHONY: clean_fits
clean_fits:
	rm -r $(fits) $(debug_plots) || true
	make clean

# but if they really want to nuke everything too they can
.PHONY: clean_all
clean_all:
	rm -r $(all_my_dirs) || true

$(all_my_dirs):
	mkdir $@

$(catalogs): $(catalog_script)
	python $(catalog_script) $@

# First make a preliminary list of stars, which we'll then give to the user to
# sort through to make the final list
# The star lists require the catalogs, as we need to exclude anything that's
# one of the clusters from our selection of stars.
# To do this we use SECONDEXPANSION, and turn the star list into a catalog name
.SECONDEXPANSION:
$(v1_star_lists): %: | $(v1_star_list_script) $$(dir %)$$(cat)
	python $(v1_star_list_script) $@ $(dir $@)$(cat) $(psf_pixel_size)

# Creating the actual list of PSF stars requires user input.
# note that there's no dependency on the script itself here. That's becuase I
# don't want any unimportant future changes to make me redo all the star
# selection, since it's tedious. If I need to remake these, just delete the
# files
.SECONDEXPANSION:
$(psf_star_lists): %: | $$(dir %)$$(star_prelim)
	python $(psf_star_list_script) $@ $(dir $@)$(star_prelim) $(psf_pixel_size)

# we use SECONDEXPANSION to parametrize over all clusters
# The PSF creation depends on the PSF star lists. If we made our own this needs
# to depend on that
.SECONDEXPANSION:
$(psfs_my): %: $(psf_creation_script) $$(dir %)$$(star_psf)
	python $(psf_creation_script) $@ $(psf_oversampling_factor) $(psf_pixel_size) my

.SECONDEXPANSION:
$(psfs_legus): %: $(psf_creation_script)
	python $(psf_creation_script) $@ $(psf_oversampling_factor) $(psf_pixel_size) legus

# Then the plot comparing the psfs
.SECONDEXPANSION:
$(psf_comp_plots): %: $$(dir %)$$(psf_legus) $$(dir %)$$(psf_my) $$(dir %)$$(star_psf) $(psf_comparison_script)
	python $(psf_comparison_script) $@ $(psf_oversampling_factor) $(psf_pixel_size)

# And the demo image that will go in the paper
$(psf_demo_image): $(fit_psfs) $(psf_demo_image_script)
	python $(psf_demo_image_script) $@ $(psf_oversampling_factor) $(fit_psfs)

# The first step in the fitting is the creation of the sigma image
$(sigma_images): $(sigma_script)
	python $(sigma_script) $@

# We only mask regions near clusters to save time, so we need the catalogs
# to do the masking
$(masks): %: $(mask_script) $$(dir %)$$(cat) $$(dir %)$$(sigma_image)
	python $(mask_script) $@ $(dir $@)$(cat) $(dir $@)$(sigma_image)

# Then we can actually do the fitting, which depends on
# the sigma image, psf, mask, and cluster catalog.
.SECONDEXPANSION:
$(fits): %: $(fitting_script) $(fit_utils) $$(dir %)$$(fit_psf) $$(dir %)$$(sigma_image) $$(dir %)$$(mask) $$(dir %)$$(cat)
	python $(fitting_script) $@ $(dir $@)$(fit_psf) $(psf_oversampling_factor) $(dir $@)$(sigma_image) $(dir $@)$(mask) $(dir $@)$(cat) $(fit_region_size)

.SECONDEXPANSION:
$(fits_ryon): %: $(fitting_script) $(fit_utils) $$(dir %)$$(fit_psf) $$(dir %)$$(sigma_image) $$(dir %)$$(mask) $$(dir %)$$(cat)
	python $(fitting_script) $@ $(dir $@)$(fit_psf) $(psf_oversampling_factor) $(dir $@)$(sigma_image) $(dir $@)$(mask) $(dir $@)$(cat) $(fit_region_size) ryon_like

# Add the derived properties to these catalogs
.SECONDEXPANSION:
$(final_cats): %: $(final_catalog_script) $(fit_utils) $$(dir %)$$(fit) $$(dir %)$$(fit_psf) $$(dir %)$$(sigma_image) $$(dir %)$$(mask)
	python $(final_catalog_script) $@ $(dir $@)$(fit) $(dir $@)$(fit_psf) $(psf_oversampling_factor) $(dir $@)$(sigma_image) $(dir $@)$(mask) $(fit_region_size)

.SECONDEXPANSION:
$(final_cats_ryon): %: $(final_catalog_script) $(fit_utils) $$(dir %)$$(fit_ryon) $$(dir %)$$(fit_psf) $$(dir %)$$(sigma_image) $$(dir %)$$(mask)
	python $(final_catalog_script) $@ $(dir $@)$(fit_ryon) $(dir $@)$(fit_psf) $(psf_oversampling_factor) $(dir $@)$(sigma_image) $(dir $@)$(mask) $(fit_region_size) ryon_like

# Make the comparisons to Ryon+17's results and other comparison plots
$(comparison_plot): $(comparison_script) $(final_cats) $(final_cats_ryon)
	python $(comparison_script) $@ $(final_cats_ryon) $(final_cats)

$(param_dist_plot): $(parameters_dist_script) $(final_cats)
	python $(parameters_dist_script) $@ $(final_cats)

$(all_galaxies_plot): $(final_cats) $(all_galaxies_script)
	python $(all_galaxies_script) $@ $(final_cats)

$(all_galaxies_iso_plot): $(final_cats) $(all_galaxies_iso_script)
	python $(all_galaxies_iso_script) $@ $(final_cats)

$(stacked_distribution_plot): $(final_cats) $(stacked_distribution_script)
	python $(stacked_distribution_script) $@ $(final_cats)

$(dynamical_age_plot) $(bound_fraction_plot) &: $(final_cats) $(dynamical_age_script)
	python $(dynamical_age_script) $(dynamical_age_plot) $(bound_fraction_plot) $(final_cats)

$(density_plot) $(density_fits_txt) &: $(final_cats) $(density_script)
	python $(density_script) $(density_plot) $(density_fits_txt) $(final_cats)

$(toy_model_plot): $(toy_model_script) $(mass_radius_table)
	python $(toy_model_script) $@ $(mass_radius_table) $(final_cats)

$(example_fit_plot): $(final_cats) $(example_plot_script)
	python $(example_plot_script) $@ $(psf_oversampling_factor) $(psf_pixel_size) $(fit_region_size)

$(fit_quality_plot): $(final_cats) $(fit_quality_script)
	python $(fit_quality_script) $@ $(run_name) $(final_cats)

$(galaxy_table): $(final_cats) $(galaxy_table_script)
	python $(galaxy_table_script) $@ $(psf_oversampling_factor) $(psf_pixel_size) $(psf_type) $(final_cats)

# Various mass-radius relation plots
# need make v4.3 for this to work (can be installed with conda)
$(mass_radius_legus_full_plot) $(mass_radius_legus_full_txt) &: $(final_cats) $(mass_radius_legus_full_script) $(mass_radius_utils) $(mass_radius_utils_mle_fitting) $(mass_radius_utils_plotting)
	python $(mass_radius_legus_full_script) $(mass_radius_legus_full_plot) $(mass_radius_legus_full_txt) $(final_cats)

$(mass_radius_legus_young_plot) $(mass_radius_legus_young_txt) &: $(final_cats) $(mass_radius_legus_young_script) $(mass_radius_utils) $(mass_radius_utils_mle_fitting) $(mass_radius_utils_plotting)
	python $(mass_radius_legus_young_script) $(mass_radius_legus_young_plot) $(mass_radius_legus_young_txt) $(final_cats)

$(mass_radius_legus_agesplit_plot) $(mass_radius_legus_agesplit_txt) &: $(final_cats) $(mass_radius_legus_agesplit_script) $(mass_radius_utils) $(mass_radius_utils_mle_fitting) $(mass_radius_utils_plotting)
	python $(mass_radius_legus_agesplit_script) $(mass_radius_legus_agesplit_plot) $(mass_radius_legus_agesplit_txt) $(final_cats)

$(mass_radius_legus_ssfrsplit_plot) $(mass_radius_legus_ssfrsplit_txt) &: $(final_cats) $(mass_radius_legus_ssfrsplit_script) $(mass_radius_utils) $(mass_radius_utils_mle_fitting) $(mass_radius_utils_plotting)
	python $(mass_radius_legus_ssfrsplit_script) $(mass_radius_legus_ssfrsplit_plot) $(mass_radius_legus_ssfrsplit_txt) $(final_cats)

$(mass_radius_legus_mw_txt) &: $(final_cats) $(mass_radius_legus_mw_script) $(mass_radius_utils) $(mass_radius_utils_mle_fitting) $(mass_radius_utils_external_data)
	python $(mass_radius_legus_mw_script) $(mass_radius_legus_mw_txt) $(final_cats)

$(mass_radius_legus_external_txt) &: $(final_cats) $(mass_radius_legus_external_script) $(mass_radius_utils) $(mass_radius_utils_mle_fitting) $(mass_radius_utils_external_data)
	python $(mass_radius_legus_external_script) $(mass_radius_legus_external_txt) $(final_cats)

$(mass_radius_legus_mw_external_plot) $(mass_radius_legus_mw_external_txt) &: $(final_cats) $(mass_radius_legus_mw_external_script) $(mass_radius_utils) $(mass_radius_utils_plotting) $(mass_radius_utils_mle_fitting) $(mass_radius_utils_external_data)
	python $(mass_radius_legus_mw_external_script) $(mass_radius_legus_mw_external_plot) $(mass_radius_legus_mw_external_txt) $(final_cats)

# combine all tables into the final one
$(mass_radius_table): $(mass_radius_final_table_script) $(mass_radius_legus_full_txt) $(mass_radius_legus_young_txt) $(mass_radius_legus_agesplit_txt) $(mass_radius_legus_mw_txt)  $(mass_radius_legus_external_txt)  $(mass_radius_legus_mw_external_txt)
	python $(mass_radius_final_table_script) $(mass_radius_table) $(mass_radius_legus_full_txt) $(mass_radius_legus_young_txt) $(mass_radius_legus_agesplit_txt) $(mass_radius_legus_mw_txt)  $(mass_radius_legus_external_txt)  $(mass_radius_legus_mw_external_txt)

# ------------------------------------------------------------------------------
#  Artificial cluster tests
# ------------------------------------------------------------------------------
# Set up the artificial cluster tests separately. This is a bit clunky as it
# isn't as automated as the normal runs, but that's okay as it's a different
# workflow from those other runs, slightly.
# the artificial image needs the long name so the code can find it
base_field = ngc628-c
artificial_catalog = $(artificial_dir)true_catalog.txt
artificial_image = $(artificial_dir)hlsp_legus_hst_acs_artificial_f555w_v1_drc.fits
artificial_psf = $(artificial_dir)$(my_dirname)$(psf_my)
artificial_sigma_image = $(artificial_dir)$(my_dirname)$(sigma_image)
artificial_mask_image = $(artificial_dir)$(my_dirname)$(mask)
artificial_fit = $(artificial_dir)$(my_dirname)$(fit)
artificial_final_cat = $(artificial_dir)$(my_dirname)$(final_cat)

# use the psf from another galaxy
$(artificial_psf): $(psfs_my)
	cp $(data_home)/$(base_field)/$(my_dirname)$(psf_my) $@

# Make the catalog with the true locations and parameters of the clusters
# This depends on the final catalogs so I can sample realistic clusters
$(artificial_catalog): $(final_cats) $(artificial_cluster_catalog_script)
	python $(artificial_cluster_catalog_script) $@ $(base_field) $(final_cats)

# the artificial image with fake clusters
$(artificial_image): $(artificial_cluster_image_script) $(artificial_catalog)
	python $(artificial_cluster_image_script) $@ $(artificial_catalog) $(psf_oversampling_factor) $(fit_region_size) $(artificial_psf) $(base_field)

# sigma image is done with the normal pipeline. It does depend on the original image
$(artificial_sigma_image): $(sigma_script) $(artificial_image)
	python $(sigma_script) $@

# mask is done automatically
$(artificial_mask_image): $(mask_script) $(artificial_catalog) $(artificial_sigma_image) $(artificial_image)
	python $(mask_script) $@ $(artificial_catalog) $(artificial_sigma_image)

# then we can do the fitting and postprocessing!
$(artificial_fit): %: $(fitting_script) $(fit_utils) $(artificial_psf) $(artificial_sigma_image) $(artificial_mask_image) $(artificial_catalog)
	python $(fitting_script) $@ $(artificial_psf) $(psf_oversampling_factor) $(artificial_sigma_image) $(artificial_mask_image) $(artificial_catalog) $(fit_region_size)

# Add the derived properties to these catalogs
$(artificial_final_cat): $(final_catalog_script) $(fit_utils) $(artificial_fit) $(artificial_psf) $(artificial_sigma_image) $(artificial_mask_image)
	python $(final_catalog_script) $@ $(artificial_fit) $(artificial_psf) $(psf_oversampling_factor) $(artificial_sigma_image) $(artificial_mask_image) $(fit_region_size)

$(artificial_comparison): $(artificial_final_cat) $(artificial_comparison_script)
	python $(artificial_comparison_script) $@ $(artificial_final_cat) $(artificial_psf) $(psf_oversampling_factor)
