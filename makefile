data_home = /Users/gillenb/google_drive/research/legus
data_dirs = $(data_home)/ngc1313-e \
            $(data_home)/ngc1313-w \
            $(data_home)/ngc628-c \
            $(data_home)/ngc628-e

# ------------------------------------------------------------------------------
#
# Python scripts
#
# ------------------------------------------------------------------------------
catalog_script = ./legus_sizes/format_catalogs.py


# ------------------------------------------------------------------------------
#
# Things to create
#
# ------------------------------------------------------------------------------
dir_to_catalog = $(1)/catalog.txt
all_catalogs = $(foreach dir,$(data_dirs),$(call dir_to_catalog,$(dir)))

# ------------------------------------------------------------------------------
#
#  Rules
#
# ------------------------------------------------------------------------------
# https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html

all: $(all_catalogs)

.PHONY: clean
clean:
	rm $(all_catalogs)

$(all_catalogs): $(catalog_script)
	python $(catalog_script) $@