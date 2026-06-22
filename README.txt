This repository contains code for the paper Multiwavelength Analysis of Six Luminous Fast Blue Optical Transients (https://arxiv.org/abs/2601.18926)

Each figure in the paper can be reproduced by running the corresponding .py file.  The exception is Figure 10, which is produced by the corresponding .ipynb file.
This repository also includes the raw data required for the figures and a variety of helper .py files. 

The data was produced using two separate conda environments, one for the .ipynb notebook used to produce Figure 10, and one for the .py files to produce all other figures.

List of relevant packages and software used to produce all figures EXCEPT Figure 10:
-- astro-sedpy 0.3.2
-- astropy 6.0.1
-- extinction 0.4.6
-- gphoton 1.28.9
-- numpy 1.26.4
-- python 3.11.9
-- scipy 1.13.0

List of relevant packages to produce Figure 10:
-- astro-prospector 1.14.0
-- astro-sedpy 0.3.2
-- astropy 6.1.4
-- extinction 0.4.6
-- fsps 0.4.7
-- h5py 3.12.1
-- hdf5 1.14.3
-- numpy 11.26.4
-- python 3.12.7
-- scipy 1.14.1

I also needed to add custom filters to sedpy, which are located in the filter folder of this repository.