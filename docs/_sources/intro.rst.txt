grandlig : Grand Canonical Sampling of Small Molecules in OpenMM
====================================================

Background
----------

This Python module is designed to be run with OpenMM in order to simulate grand
canonical Monte Carlo (GCMC) insertion and deletion moves of small molecules.
This allows the particle number to vary according to a fixed chemical
potential, and offers enhanced sampling of small molecules in occluded
binding sites.
The theory behind our work on GCMC sampling can be found in the References
section below.

Installation & Usage
--------------------

This module can be installed from this directory by running the following
command:

.. code:: bash

    python setup.py install


The dependencies of this module can be installed as:

.. code:: bash

    conda install -c omnia openmm mdtraj parmed openmmtools pymbar
    pip install lxml

Alternatively, grand can be installed directly, using conda:

.. code:: bash

    conda install -c omnia -c anaconda -c conda-forge -c essexlab grand

The grand module is released under the MIT licence. If results from this
module contribute to a publication, we only ask that you cite the
publications below - particularly reference 1, in which the grand module
was first presented).

Contributors
------------
- Will Poole `<wp1g16@soton.ac.uk>`
- Marley Samways `<mls2g13@soton.ac.uk>`
- Ollie Melling `<ojm2g16@soton.ac.uk>`
- Hannah Bruce Macdonald

Contact
-------

If you have any problems or questions regarding this module, please contact
one of the contributors, or send an email to `<j.w.essex@soton.ac.uk>`.

References
----------
1. M. L. Samways, H. E. Bruce Macdonald, J. W. Essex, J. Chem. Inf. Model.,
2020, 60, 4436-4441, DOI: https://doi.org/10.1021/acs.jcim.0c00648

2. O. J. Melling, M. L. Samways, Y. Ge, D. L. Mobley, J. W. Essex, J. Chem. Theory Comput., 2023,
DOI: https://doi.org/10.1021/acs.jctc.2c00823

3. G. A. Ross, M. S. Bodnarchuk, J. W. Essex, J. Am. Chem. Soc., 2015,
137, 47, 14930-14943, DOI: https://doi.org/10.1021/jacs.5b07940

4. G. A. Ross, H. E. Bruce Macdonald, C. Cave-Ayland, A. I. Cabedo
Martinez, J. W. Essex, J. Chem. Theory Comput., 2017, 13, 12, 6373-6381, DOI:
https://doi.org/10.1021/acs.jctc.7b00738