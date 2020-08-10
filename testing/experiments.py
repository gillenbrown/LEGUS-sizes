"""
experiments.py - script that holds temporary experimental things

This is set up this way so that I can easily pass in all the catalogs
"""
import sys
from pathlib import Path

import numpy as np
from astropy import table
from matplotlib import colors, cm
import betterplotlib as bpl
import cmocean

bpl.set_style()

# ======================================================================================
#
# load the parameters the user passed in
#
# ======================================================================================
sentinel_name = Path(sys.argv[1])
catalogs = []
for item in sys.argv[2:]:
    cat = table.Table.read(item, format="ascii.ecsv")
    cat["galaxy"] = Path(item).parent.parent.name
    catalogs.append(cat)
# then stack them together in one master catalog
big_catalog = table.vstack(catalogs, join_type="inner")

# ======================================================================================
#
# Experiments start here
#
# ======================================================================================


# ======================================================================================
#
# Do not remove this line!!
#
# ======================================================================================
sentinel_name.touch()
