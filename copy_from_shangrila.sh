rsync -av gillenb@shangrila.astro.lsa.umich.edu:"/u/home/gillenb/code/mine/LEGUS-sizes/plots/*" /Users/gillenb/code/legus_sizes/plots/
rsync -av --exclude '*.fits' --exclude '*tar.gz' --exclude '*.png' gillenb@shangrila.astro.lsa.umich.edu:"/u/home/gillenb/legus/*" /Users/gillenb/google_drive/research/legus/data/
rsync -av --exclude '*drc.fits' --exclude '*sci.fits' --exclude '*sigma_electrons.fits' --exclude '*tar.gz' --exclude '*bootstrapping*' gillenb@shangrila.astro.lsa.umich.edu:"/u/home/gillenb/legus/*" /Users/gillenb/google_drive/research/legus/data/
rsync -av --exclude '*drc.fits' --exclude '*sci.fits' --exclude '*sigma_electrons.fits' --exclude '*tar.gz' gillenb@shangrila.astro.lsa.umich.edu:"/u/home/gillenb/legus/*" /Users/gillenb/google_drive/research/legus/data/
