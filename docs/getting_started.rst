Getting Started
===============

Background
~~~~~~~~~~
This Python module is designed to be run with OpenMM in order to simulate
grand canonical Monte Carlo (GCMC) insertion and deletion moves of small fragment-like molecules.
This allows the particle number to vary according to a fixed chemical potential,
and offers enhanced sampling of molecules in occluded binding sites.
The theory behind our work on GCMC sampling can be found in the References section below.

Installation and Usage
~~~~~~~~~~
This module can be installed by cloning this repo and running:


.. code:: bash

    pip install .

Alternatively, grandlig can be installed via conda/mamba by the following:

.. code-block:: python

    conda install -c conda-forge grandlig


Contributors
~~~~~~~~~~~
- Will Poole: wp1g16@soton.ac.uk
- Marley Samways: mls2g13@soton.ac.uk
- Ollie Melling: ojm2g16@soton.ac.uk
- Hannah Bruce Macdonald

