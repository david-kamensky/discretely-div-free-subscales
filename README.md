# Variational multiscale modeling with discretely divergence-free subscales
This repository contains FEniCS and tIGAr-based scripts to accompany the papers
```
@article{Evans2020,
title = "Variational multiscale modeling with discretely divergence-free subscales",
journal = "Computers \& Mathematics with Applications",
volume = "80",
number = "11",
pages = "2517--2537",
year = "2020",
note = "High-Order Finite Element and Isogeometric Methods 2019",
issn = "0898-1221",
author = "J. A. Evans and D. Kamensky and Y. Bazilevs",
}

@article{Calfy2021,
title = "Variational multiscale modeling with discretely divergence-free subscales: Non-divergence-conforming discretizations",
journal = "Computers \& Mathematics with Applications",
author = "S. L. Calfy and J. A. Evans and D. Kamensky",
year = "2021",
note = "Under review."
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
