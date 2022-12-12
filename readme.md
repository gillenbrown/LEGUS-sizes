# LEGUS-sizes
This repository holds the code and the catalog used in [Brown & Gnedin 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.5935B/abstract). The catalog is the `cluster_sizes_brown_gnedin_21.txt` file here. For a detailed description see the webpage: [gillenbrown.github.io/LEGUS-sizes](https://gillenbrown.github.io/LEGUS-sizes).

## The Code
This repository holds all the code used in the paper. I used a makefile to automate everything. See that for a detailed outline of how the pipeline and analysis is done. Note that the makefile is structured so that it can be run with parallel make (e.g. `make -j4`). The analysis will be parallelized over the fields.

The `pipeline` directory holds the files that calculate the radius and generate the public catalog. This holds the bulk of the code for this project. `fit.py` and `fit_utils.py` hold the code to do the actual cluster fitting.  The `analysis` directory holds scripts used to generate many of the plots in the paper. The `docs` folder holds the webpage with the catalog description for hosting with GitHub pages.

Note that I have included the data from the [Krumholz, McKee, and Bland-Hawthorn 2019](https://ui.adsabs.harvard.edu/abs/2019ARA%26A..57..227K/abstract) review as a [submodule](https://bitbucket.org/krumholz/cluster_review). If you clone this repository, you'll need a few extra commands to also clone that data. In the repository's directory, run `git submodule init` then `git submodule update`.
