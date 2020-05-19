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
star_list_script = ./legus_sizes/star_list.py

# ------------------------------------------------------------------------------
#
# Things to create
#
# ------------------------------------------------------------------------------
dir_to_catalog = $(1)/clean_catalog.txt
all_catalogs = $(foreach dir,$(data_dirs),$(call dir_to_catalog,$(dir)))

dir_to_star_list = $(1)/psf_star_list.txt
all_star_lists = $(foreach dir,$(data_dirs),$(call dir_to_star_list,$(dir)))
star_list_to_catalog = $(subst psf_star_list.txt,clean_catalog.txt,$(1))
# ------------------------------------------------------------------------------
#
#  Rules
#
# ------------------------------------------------------------------------------
# https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html

all: $(all_catalogs) $(all_star_lists)

.PHONY: clean
clean:
	rm $(all_catalogs) $(all_star_lists)

$(all_catalogs): $(catalog_script)
	python $(catalog_script) $@

# The star lists require the catalogs, as we need to exclude anything that's
# one of the clusters from our selection of stars
# To do this we use SECONDEXPANSION, and turn the star list into a catalog name
# note that there's no dependency on the script itself here. That's becuase I
# don't want any unimportant future changes to make me redo all the star
# selection. If I need to remake these, just delete the files
.SECONDEXPANSION:
$(all_star_lists): %: $$(call star_list_to_catalog, %)
	python $(star_list_script) $@ $(call star_list_to_catalog, $@)