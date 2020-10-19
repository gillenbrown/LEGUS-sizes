rsync -av gillenb@shangrila.astro.lsa.umich.edu:"/u/home/gillenb/code/mine/LEGUS-sizes/plots*" /Users/gillenb/code/legus_sizes/
rsync -av --exclude '*.fits' --exclude '*tar.gz' --exclude '*.png' gillenb@shangrila.astro.lsa.umich.edu:"/u/home/gillenb/code/mine/LEGUS-sizes/data/*" /Users/gillenb/code/legus_sizes/data/
rsync -av --exclude '*drc.fits' --exclude '*sci.fits' --exclude '*tar.gz' --exclude '*.png' gillenb@shangrila.astro.lsa.umich.edu:"/u/home/gillenb/code/mine/LEGUS-sizes/data/*" /Users/gillenb/code/legus_sizes/data/
rsync -av --exclude '*drc.fits' --exclude '*sci.fits' --exclude '*tar.gz' gillenb@shangrila.astro.lsa.umich.edu:"/u/home/gillenb/code/mine/LEGUS-sizes/data/*" /Users/gillenb/code/legus_sizes/data/
