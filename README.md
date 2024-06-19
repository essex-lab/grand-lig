[![Build Status](https://travis-ci.org/essex-lab/grand.svg?branch=master)](https://travis-ci.org/essex-lab/grand)
[![Anaconda-Server Badge](https://anaconda.org/essexlab/grand/badges/version.svg)](https://anaconda.org/essexlab/grand)
[![Anaconda-Server Badge](https://anaconda.org/essexlab/grand/badges/downloads.svg)](https://anaconda.org/essexlab/grand)
[![Documentation Status](https://readthedocs.org/projects/grand/badge/?version=latest)](https://grand.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/270705695.svg)](https://zenodo.org/badge/latestdoi/270705695)

# From Molssi
<!-- [//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/grandlig/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/grandlig/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/grandlig/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/grandlig/branch/main) -->

# _grandlig_ : Grand Canonical Ligand Sampling in OpenMM

*PRIVATE REPO FOR DEVELOPMENT*

### Background

This Python module is designed to be run with OpenMM in order to simulate grand
canonical Monte Carlo (GCMC) and nonequilibirum candidate Monte Carlo (GCNCMC) insertion and deletion moves of ligand and water molecules.
This allows the particle number to vary according to a fixed chemical
potential, and offers enhanced sampling of molecules in occluded
binding sites.
The theory behind our work on GCMC sampling can be found in the References
section below.

### Installation & Usage

This module can be installed from this directory by running the following
command:

```commandline
python setup.py install
```

The unit tests can then be carried out by running the following command from
this directory:
```commandline
python setup.py test
```

The dependencies of this module can be installed as:

```commandline
conda install -c conda-forge -c omnia openmmtools
pip install lxml
```
Many of grand's dependencies (openmm, mdtraj, pymbar, parmed) are also dependencies of 
openmmtools, and will be installed alongside openmmtools.

Alternatively, _grandlig_ and its dependencies can be installed via conda:
```commandline
conda install -c omnia -c anaconda -c conda-forge -c essexlab grand
```

Several (very short) examples of how this module is ran alongside OpenMM can be found in
the `examples/` directory.
Additional [examples](https://github.com/essex-lab/grand-paper) and 
[documentation](https://grand.readthedocs.io/en/latest/) are also available.

The _grand_ module is released under the MIT licence. If results from this
module contribute to a publication, we ask that you cite Refs. 1 and 2, below.
Ref. 1 discusses the initial implemention while ref. 2 discusses the implementation
and testing of the non-equilibrium moves.
Additional references describing the theory upon which the GCMC implemention
in _grand_ is based are also provided below (Refs. 3-4).

- M. L. Samways, H. E. Bruce Macdonald, J. W. Essex, _J. Chem. Inf. Model._,
2020, 60, 4436-4441, DOI: https://doi.org/10.1021/acs.jcim.0c00648

Additional references describing the theory of GCMC are provided below.

### Contributors

- Will Poole `<wp1g16@soton.ac.uk>` 
- Marley Samways `<mls2g13@soton.ac.uk>`
- Ollie Melling `<ojm2g16@soton.ac.uk>`
- Hannah Bruce Macdonald

### Contact

If you have any problems or questions regarding this module, please contact
one of the contributors, or send an email to `<j.w.essex@soton.ac.uk>`.

### References

1. M. L. Samways, H. E. Bruce Macdonald, J. W. Essex, _J. Chem. Inf. Model._,
2020, 60, 4436-4441, DOI: https://doi.org/10.1021/acs.jcim.0c00648
2. O. J. Melling, M. L. Samways, Y. Ge, D. L. Mobley, J. W. Essex, _J. Chem. Theory Comput._, 2023,
DOI: https://doi.org/10.1021/acs.jctc.2c00823
3. G. A. Ross, M. S. Bodnarchuk, J. W. Essex, _J. Am. Chem. Soc._, 2015,
137, 47, 14930-14943, DOI: https://doi.org/10.1021/jacs.5b07940
4. G. A. Ross, H. E. Bruce Macdonald, C. Cave-Ayland, A. I. Cabedo
Martinez, J. W. Essex, _J. Chem. Theory Comput._, 2017, 13, 12, 6373-6381, DOI:
https://doi.org/10.1021/acs.jctc.7b00738


### Copyright

Copyright (c) 2024, Will Poole


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.

