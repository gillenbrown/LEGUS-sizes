# LEGUS-sizes
This repository holds the code and the catalog used in Brown & Gnedin 2021. The catalog is the `cluster_sizes_brown_gnedin_21.txt` file here. For a detailed description see the webpage: [gillenbrown.github.io/LEGUS-sizes](https://gillenbrown.github.io/LEGUS-sizes).

## The Code
This repository holds all the code used in the paper. I used a makefile to automate everything. See that for a detailed outline of how the pipeline and analysis is done.

What follows is a simplified description of the code. The `pipeline` directory holds the files that calculate the radius and generate the public catalog. This holds the bulk of the code for this project. `fit.py` and `fit_utils.py` hold the code to do the actual cluster fitting.  The `analysis` directory holds scripts used to generate many of the plots in the paper. Lastly, the `testing` directory holds some Jupyter notebooks with various tests or simple analysis. Some plots in the paper are generated here. The `docs` folder holds the webpage with the catalog description for hosting with GitHub pages.
