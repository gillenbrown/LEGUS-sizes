rsync -av /Users/gillenb/code/legus_sizes/plots* gillenb@shangrila.astro.lsa.umich.edu:"/u/home/gillenb/code/mine/LEGUS-sizes/plots/"
rsync -av --exclude '*drc.fits' --exclude '*tar.gz' --exclude '*padagb-mwext-avgapcor*' --exclude '*/cluster_fit_plots/*.png' /Users/gillenb/code/legus_sizes/data/* gillenb@shangrila.astro.lsa.umich.edu:/u/home/gillenb/code/mine/LEGUS-sizes/data/
