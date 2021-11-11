# SpiralGalaxies

The routines in this repository populate points, representing star clusters, 
in a spiral arm distribution to mimic the young star cluster populations in spiral galaxies. This can then be used to compute the Two-Point Correlation Function (TPCF)
of the star clusters, to explore the effects of the large-scale spiral distribution in the galaxy on the TPCF in real observations. Toy models of distributions, 
such as those described in this repository provides significant insight into the observed TPCFs (see [Menon et al 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.507.5542M/abstract)
for a study where this is demonstrated). 

The jupyter notebook `tutorial.ipynb` contains a basic tutorial of the functionality offered here. 

### Requirements
The code in this repo requires the following python libraries - 
1. [scikit-learn](https://scikit-learn.org/stable/index.html) : Used for quick counting of pairs in the TPCF. Perform `pip install -U scikit-learn` if you don't have this. 
2. [joblib](https://joblib.readthedocs.io/en/latest/index.html) : Used for parallelising the TPCF computation, i.e. makes things faster. Perform `pip install --user joblib` if you don't. 

### Contact
Email me at shyam.menon@anu.edu.au with any questions. 
