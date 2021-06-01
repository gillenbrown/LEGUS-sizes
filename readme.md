# LEGUS-sizes
This repository holds the code and the catalog used in Brown & Gnedin 2021. The catalog is the `cluster_sizes_brown_gnedin_21.txt` file here. For a detailed description see the webpage: [gillenbrown.github.io/LEGUS-sizes](https://gillenbrown.github.io/LEGUS-sizes).

## The Code
This repository holds all the code used in the paper. I used a makefile to automate everything. See that for a detailed outline of how the pipeline and analysis is done.

The `pipeline` directory holds the files that calculate the radius and generate the public catalog. This holds the bulk of the code for this project. `fit.py` and `fit_utils.py` hold the code to do the actual cluster fitting.  The `analysis` directory holds scripts used to generate many of the plots in the paper. The `docs` folder holds the webpage with the catalog description for hosting with GitHub pages.

Note that I have included the data from the [Krumholz, McKee, and Bland-Hawthorn 2019](https://ui.adsabs.harvard.edu/abs/2019ARA%26A..57..227K/abstract) review as a [submodule](https://bitbucket.org/krumholz/cluster_review). If you clone this repository, you'll need a few extra commands to also clone that data. In the repository's directory, run `git submodule init` then `git submodule update`.  