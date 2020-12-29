rsync -av /Users/gillenb/code/legus_sizes/outputs_* gillenb@shangrila.astro.lsa.umich.edu:"/u/home/gillenb/code/mine/LEGUS-sizes/"
rsync -av --exclude '*drc.fits' --exclude '*tar.gz' --exclude '*padagb-mwext-avgapcor*' --exclude '*/cluster_fit_plots/*.png' /Users/gillenb/code/legus_sizes/data/* gillenb@shangrila.astro.lsa.umich.edu:/u/home/gillenb/code/mine/LEGUS-sizes/data/
#rsync -av /Users/gillenb/code/legus_sizes/mcmc_chain*.h5 gillenb@shangrila.astro.lsa.umich.edu:"/u/home/gillenb/code/mine/LEGUS-sizes/"
