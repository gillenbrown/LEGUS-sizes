data_home = /Users/gillenb/google_drive/research/legus
data_dirs = $(data_home)/data/ngc1313-e

#data_home = /Users/gillenb/google_drive/research/legus/data
## This directory should have nothing but directories with data
## We'll do this complicated line that just gets all directories inside data_home
#data_dirs = $(sort $(dir $(wildcard $(data_home)/*/)))
# ------------------------------------------------------------------------------
#
# Python scripts
#
# ------------------------------------------------------------------------------
catalog_script = ./legus_sizes/format_catalogs.py
v1_star_list_script = ./legus_sizes/preliminary_star_list.py
psf_star_list_script = ./legus_sizes/select_psf_stars.py

# ------------------------------------------------------------------------------
#
# Cleaned cluster catalogs
#
# ------------------------------------------------------------------------------
dir_to_catalog = $(1)/clean_catalog.txt
all_catalogs = $(foreach dir,$(data_dirs),$(call dir_to_catalog,$(dir)))

# ------------------------------------------------------------------------------
#
# List of stars eligible to be put into the PSF
#
# ------------------------------------------------------------------------------
dir_to_v1_star_list = $(1)/preliminary_stars.txt
all_v1_star_lists = $(foreach dir,$(data_dirs),$(call dir_to_v1_star_list,$(dir)))
v1_star_list_to_catalog = $(subst preliminary_stars.txt,clean_catalog.txt,$(1))

# ------------------------------------------------------------------------------
#
# User-selected stars to make the PSF
#
# ------------------------------------------------------------------------------
dir_to_psf_star_list = $(1)/psf_stars.txt
all_psf_star_lists = $(foreach dir,$(data_dirs),$(call dir_to_psf_star_list,$(dir)))
psf_star_list_to_v1_list = $(subst psf_stars.txt,preliminary_stars.txt,$(1))

# ------------------------------------------------------------------------------
#
#  Rules
#
# ------------------------------------------------------------------------------
# https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html

all: $(all_catalogs) $(all_psf_star_lists)

.PHONY: clean
clean:
	rm $(all_catalogs) $(all_v1_star_lists) $(all_psf_star_lists)

$(all_catalogs): $(catalog_script)
	python $(catalog_script) $@

# First make a preliminary list of stars, which we'll then give to the user to
# sort through to make the final list
# The star lists require the catalogs, as we need to exclude anything that's
# one of the clusters from our selection of stars.
# To do this we use SECONDEXPANSION, and turn the star list into a catalog name
.SECONDEXPANSION:
$(all_v1_star_lists): %: $$(call v1_star_list_to_catalog, %) $(v1_star_list_script)
	python $(v1_star_list_script) $@ $(call v1_star_list_to_catalog, $@)

# Creating the actual list of PSF stars requires user input.
# note that there's no dependency on the script itself here. That's becuase I
# don't want any unimportant future changes to make me redo all the star
# selection, since it's tedious. If I need to remake these, just delete the
# files
.SECONDEXPANSION:
$(all_psf_star_lists): %: $$(call psf_star_list_to_v1_list, %)
	python $(psf_star_list_script) $@ $(call psf_star_list_to_v1_list, $@)
