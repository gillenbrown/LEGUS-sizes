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
# Python scripts
#
# ------------------------------------------------------------------------------
catalog_script = ./legus_sizes/format_catalogs.py
psf_star_list_script = ./legus_sizes/select_psf_stars.py
psf_creation_script = ./legus_sizes/make_psf.py
sigma_script = ./legus_sizes/make_sigma_image.py
fitting_script = ./legus_sizes/fit.py
final_catalog_script = ./legus_sizes/derived_properties.py
final_catalog_script_no_mask = ./legus_sizes/derived_properties_ryon.py
comparison_script = ./legus_sizes/ryon_comparison.py
radii_def_plot_script = ./legus_sizes/radii_def_comp_plot.py
parameters_dist_script = ./legus_sizes/parameter_distribution.py
all_fields_script = ./legus_sizes/all_fields_hist.py

# ------------------------------------------------------------------------------
#
# Configuration variables
#
# ------------------------------------------------------------------------------
psf_pixel_size = 9
psf_oversampling_factor = 2
fit_region_size = 30

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
local_plots_dir = ./plots/
all_my_dirs = $(my_dirs) $(cluster_fit_dirs) $(cluster_plot_dirs) $(local_plots_dir)

# ------------------------------------------------------------------------------
#
# All the filenames I'll produce
#
# ------------------------------------------------------------------------------
cat = clean_catalog.txt
psf = psf_$(psf_oversampling_factor).fits
sigma_image = sigma_electrons.fits
fit = cluster_fits_$(fit_region_size).h5
fit_no_mask = cluster_fits_no_masking_size_$(fit_region_size).h5
final_cat = final_catalog_$(fit_region_size).txt
final_cat_no_mask = final_catalog_no_masking_$(fit_region_size).txt

# ------------------------------------------------------------------------------
#
# Put all these files in each of these directories
#
# ------------------------------------------------------------------------------
catalogs = $(foreach dir,$(my_dirs),$(dir)$(cat))
psfs = $(foreach dir,$(my_dirs),$(dir)$(psf))
sigma_images = $(foreach dir,$(my_dirs),$(dir)$(sigma_image))
fits = $(foreach dir,$(my_dirs),$(dir)$(fit))
fits_no_mask = $(foreach dir,$(my_dirs_ryon),$(dir)$(fit_no_mask))
final_cats = $(foreach dir,$(my_dirs),$(dir)$(final_cat))
final_cats_no_mask = $(foreach dir,$(my_dirs_ryon),$(dir)$(final_cat_no_mask))

# ------------------------------------------------------------------------------
#
# Various plots that will be here in this directory
#
# ------------------------------------------------------------------------------
comparison_plot = $(local_plots_dir)comparison_plot_size_$(fit_region_size).png
radii_def_comp_plot = $(local_plots_dir)radii_def_comp_plot_$(fit_region_size).png
param_dist_plot = $(local_plots_dir)parameter_distribution_size_$(fit_region_size).png
param_dist_plot_no_mask = $(local_plots_dir)parameter_distribution_ryon_galaxies_size_$(fit_region_size).png
all_fields_hist_plot = $(local_plots_dir)all_fields_$(fit_region_size).png
plots = $(comparison_plot) $(radii_def_comp_plot) $(param_dist_plot) $(param_dist_plot_no_mask) $(all_fields_hist_plot)

# ------------------------------------------------------------------------------
#
#  Rules
#
# ------------------------------------------------------------------------------
# https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html

all: $(all_my_dirs) $(plots)

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

# we use SECONDEXPANSION to parametrize over all clusters
# The PSF creation depends on the PSF star lists
.SECONDEXPANSION:
$(psfs): %: $(psf_creation_script)
	python $(psf_creation_script) $@ $(psf_oversampling_factor) $(psf_pixel_size)

# The first step in the fitting is the creation of the sigma image
$(sigma_images): $(sigma_script)
	python $(sigma_script) $@

# Then we can actually do the fitting, which depends on
# the sigma image, psf, and cluster catalog. First we remove all the debug plots
# associated with this run, as we don't know how many bootstrap iterations
# will be needed, and we want to make sure all plots come from the same run.
to_rm_debug_plots =  $(1)cluster_fit_plots/*size_$(fit_region_size)_$(2)*
.SECONDEXPANSION:
$(fits): %: $(fitting_script) $$(dir %)$$(psf) $$(dir %)$$(sigma_image) $$(dir %)$$(cat)
	rm $(call to_rm_debug_plots,$(dir $@),final) || true
	python $(fitting_script) $@ $(dir $@)$(psf) $(psf_oversampling_factor) $(dir $@)$(sigma_image) $(dir $@)$(cat) $(fit_region_size)

# for the no masking case we pass an extra parameter
$(fits_no_mask): %: $(fitting_script) $$(dir %)$$(psf) $$(dir %)$$(sigma_image) $$(dir %)$$(cat)
	rm $(call to_rm_debug_plots,$(dir $@),ryon_like) || true
	python $(fitting_script) $@ $(dir $@)$(psf) $(psf_oversampling_factor) $(dir $@)$(sigma_image) $(dir $@)$(cat) $(fit_region_size) ryon_like

# Add the derived properties to these catalogs
.SECONDEXPANSION:
$(final_cats): %: $(final_catalog_script) $$(dir %)$$(fit)
	python $(final_catalog_script) $@ $(dir $@)$(fit)

.SECONDEXPANSION:
$(final_cats_no_mask): %: $(final_catalog_script_no_mask) $$(dir %)$$(fit_no_mask)
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