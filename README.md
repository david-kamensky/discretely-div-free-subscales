# Variational multiscale modeling with weakly divergence-free subscales
This repository contains FEniCS and tIGAr-based scripts to accompany the papers
```
@article{Evans2019a,
title = "Variational multiscale modeling with discretely divergence-free subscales",
journal = "Computers and Mathematics with Applications",
author = "J. A. Evans and D. Kamensky and Y. Bazilevs",
year = "2019",
note = "In preparation."
}

@article{Evans2019b,
title = "Analysis of the method of weakly divergence-free subscales for non-divergence conforming discretizations",
journal = "Mathematical Models and Methods in Applied Sciences",
author = "J. A. Evans and D. Kamensky",
year = "2019",
note = "In preparation."
}
```
Usage requires [FEniCS](https://fenicsproject.org/), version 2019.1.  Isogeometric examples also require [tIGAr](https://github.com/david-kamensky/tIGAr).  Installation information for tIGAr can be found in the linked repository's README file.  Some examples require [TSFC](https://doi.org/10.1137/17M1130642), which can be installed for FEniCS as follows:
```
$ sudo pip3 install git+https://github.com/blechta/tsfc.git@2018.1.0
$ sudo pip3 install git+https://github.com/blechta/COFFEE.git@2018.1.0
$ sudo pip3 install git+https://github.com/blechta/FInAT.git@2018.1.0
$ sudo pip3 install singledispatch networkx pulp
```
For more detailed information, refer to comments within the provided scripts.  
