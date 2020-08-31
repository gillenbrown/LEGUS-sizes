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
data_dirs = $(sort $(dir $(wildcard $(data_home)/*/)))

# then filter out the ones that Ryon used, to compare to those
ryon_dirs = $(filter %ngc1313-e/ %ngc1313-w/ %ngc628-c/ %ngc628-e/, $(data_dirs))

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
run_name = radial_weighting_1pix

# ------------------------------------------------------------------------------
#
# Python scripts
#
# ------------------------------------------------------------------------------
catalog_script = ./pipeline/format_catalogs.py
v1_star_list_script = ./pipeline/preliminary_star_list.py
psf_star_list_script = ./pipeline/select_psf_stars.py
psf_creation_script = ./pipeline/make_psf.py
psf_comparison_script = ./analysis/psf_compare.py
psf_demo_image_script = ./analysis/psf_demo_image.py
sigma_script = ./pipeline/make_sigma_image.py
mask_script = ./pipeline/make_mask_image.py
fitting_script = ./pipeline/fit.py
fit_utils = ./pipeline/fit_utils.py
final_catalog_script = ./pipeline/derived_properties.py
final_catalog_script_no_mask = ./pipeline/derived_properties_ryon.py
comparison_script = ./analysis/ryon_comparison.py
radii_def_plot_script = ./analysis/radii_def_comp_plot.py
parameters_dist_script = ./analysis/parameter_distribution.py
all_fields_script = ./analysis/all_fields_hist.py
mass_size_script = ./analysis/mass_size.py
example_plot_script = ./analysis/example_fit.py
fit_quality_script = ./analysis/fit_quality.py
experiment_script = ./testing/experiments.py

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
local_plots_dir = ./plots_$(run_name)/
all_my_dirs = $(my_dirs) $(cluster_fit_dirs) $(cluster_plot_dirs) $(local_plots_dir)

# ------------------------------------------------------------------------------
#
# All the filenames I'll produce
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
fit_no_mask = cluster_fits_$(run_name)_no_masking_size_$(fit_region_size)_pixels_psf_$(psf_type)_stars_$(psf_pixel_size)_pixels_$(psf_oversampling_factor)x_oversampled.h5
final_cat = final_catalog_$(run_name)_$(fit_region_size)_pixels_psf_$(psf_type)_stars_$(psf_pixel_size)_pixels_$(psf_oversampling_factor)x_oversampled.txt
final_cat_no_mask = final_catalog_$(run_name)_no_masking_$(fit_region_size)_pixels_psf_$(psf_type)_stars_$(psf_pixel_size)_pixels_$(psf_oversampling_factor)x_oversampled.txt

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
fits_no_mask = $(foreach dir,$(my_dirs_ryon),$(dir)$(fit_no_mask))
final_cats = $(foreach dir,$(my_dirs),$(dir)$(final_cat))
final_cats_no_mask = $(foreach dir,$(my_dirs_ryon),$(dir)$(final_cat_no_mask))

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
# Various plots that will be here in this directory
#
# ------------------------------------------------------------------------------
psf_demo_image = $(local_plots_dir)psf_demo_$(psf_type)_stars_$(psf_pixel_size)_pixels_$(psf_oversampling_factor)x_oversampled.pdf
comparison_plot = $(local_plots_dir)comparison_plot_size_$(fit_region_size).png
radii_def_comp_plot = $(local_plots_dir)radii_def_comp_plot_$(fit_region_size).png
param_dist_plot = $(local_plots_dir)parameter_distribution_size_$(fit_region_size).png
param_dist_plot_no_mask = $(local_plots_dir)parameter_distribution_ryon_galaxies_size_$(fit_region_size).png
all_fields_hist_plot = $(local_plots_dir)all_fields_$(fit_region_size).png
mass_size_plot = $(local_plots_dir)mass_size_relation_$(fit_region_size).png
example_fit_plot = $(local_plots_dir)example_fit.png
fit_quality_plot = $(local_plots_dir)fit_quality.png
plots = $(psf_demo_image) $(psf_comp_plots) $(comparison_plot)\
        $(radii_def_comp_plot) $(param_dist_plot) $(param_dist_plot_no_mask) \
        $(all_fields_hist_plot) $(mass_size_plot) $(fit_quality_plot)
experiments_sentinel = ./testing/experiments_done.txt

# ------------------------------------------------------------------------------
#
#  Rules
#
# ------------------------------------------------------------------------------
# https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html

all: $(all_my_dirs) $(plots) $(experiments_sentinel)

# When we clean we will only clean the things after the fitting, since that
# takes so long. The "or true" thing there stops make from throwing an error
# if the files we try to remove don't exist
.PHONY: clean
clean:
	rm $(final_cats) $(final_cats_no_mask) $(plots) || true

# have another option to remove the fits too, just not the PSFs. Be careful
# that we only remove the plots for this image size.
debug_plots = $(foreach dir,$(cluster_fit_dirs),$(dir)/*size_$(fit_region_size)*)
.PHONY: clean_fits
clean_fits:
	rm -r $(fits) $(fits_no_mask) $(debug_plots) || true
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
# the sigma image, psf, mask, and cluster catalog. First we remove all the debug
# plots associated with this run, as we don't know how many bootstrap iterations
# will be needed, and we want to make sure all plots come from the same run.
to_rm_debug_plots =  $(1)cluster_fit_plots/*size_$(fit_region_size)_$(2)*
.SECONDEXPANSION:
$(fits): %: | $(fitting_script) $(fit_utils) $$(dir %)$$(fit_psf) $$(dir %)$$(sigma_image) $$(dir %)$$(mask) $$(dir %)$$(cat)
	python $(fitting_script) $@ $(dir $@)$(fit_psf) $(psf_oversampling_factor) $(dir $@)$(sigma_image) $(dir $@)$(mask) $(dir $@)$(cat) $(fit_region_size)

# for the no masking case we pass an extra parameter
$(fits_no_mask): %: | $(fitting_script) $(fit_utils) $$(dir %)$$(fit_psf) $$(dir %)$$(sigma_image) $$(dir %)$$(mask) $$(dir %)$$(cat)
	python $(fitting_script) $@ $(dir $@)$(fit_psf) $(psf_oversampling_factor) $(dir $@)$(sigma_image) $(dir $@)$(mask) $(dir $@)$(cat) $(fit_region_size) ryon_like

# Add the derived properties to these catalogs
.SECONDEXPANSION:
$(final_cats): %: | $(final_catalog_script) $(fit_utils) $$(dir %)$$(fit) $$(dir %)$$(fit_psf) $$(dir %)$$(sigma_image) $$(dir %)$$(mask)
	python $(final_catalog_script) $@ $(dir $@)$(fit) $(dir $@)$(fit_psf) $(psf_oversampling_factor) $(dir $@)$(sigma_image) $(dir $@)$(mask) $(fit_region_size)

.SECONDEXPANSION:
$(final_cats_no_mask): %: | $(final_catalog_script_no_mask) $$(dir %)$$(fit_no_mask)
	python $(final_catalog_script_no_mask) $@ $(dir $@)$(fit_no_mask)

# Make the comparison to Ryon+17's results
$(comparison_plot): $(comparison_script) $(final_cats_no_mask)
	python $(comparison_script) $@ $(final_cats_no_mask)

$(radii_def_comp_plot): $(radii_def_plot_script) $(final_cats_no_mask)
	python $(radii_def_plot_script) $@ $(final_cats_no_mask)

$(param_dist_plot): $(parameters_dist_script) $(final_cats)
	python $(parameters_dist_script) $@ $(final_cats)

$(param_dist_plot_no_mask): $(parameters_dist_script) $(final_cats_no_mask)
	python $(parameters_dist_script) $@ $(final_cats_no_mask)

$(all_fields_hist_plot): $(final_cats) $(all_fields_script)
	python $(all_fields_script) $@ $(final_cats)

$(mass_size_plot): $(final_cats) $(mass_size_script)
	python $(mass_size_script) $@  $(psf_oversampling_factor) $(psf_pixel_size) $(psf_type) $(final_cats)

$(example_fit_plot): $(final_cats) $(example_plot_script)
	python $(example_plot_script) $@ $(psf_oversampling_factor) $(psf_pixel_size) $(fit_region_size)

$(fit_quality_plot): $(final_cats) $(fit_quality_script)
	python $(fit_quality_script) $@ $(run_name) $(final_cats)

$(experiments_sentinel): $(final_cats) $(experiment_script)
	python $(experiment_script) $@ $(final_cats)