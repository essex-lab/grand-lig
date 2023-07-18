"""
Setup script to facilitate installation of the GCMC OpenMM scripts

William G. Poole
Marley L. Samways
"""

import os
from setuptools import setup


# Read version number from the __init__.py file
with open(os.path.join('grandlig', '__init__.py'), 'r') as f:
    for line in f.readlines():
        if '__version__' in line:
            version = line.split()[-1].strip('"')

setup(name="grandlig",
      version=version,
      description="OpenMM-based implementation of grand canonical Monte Carlo (GCMC) for small molecules.",
      author="William G. Poole",
      author_email="wp1g16@soton.ac.uk",
      packages=["grandlig", "grandlig.tests"],
      #install_requires=["numpy", "mdtraj"],
      setup_requires=["pytest-runner"],
      tests_require=["pytest"],
      test_suite="grandlig.tests",
      package_data={"grandlig": ["data/*", "data/tests/*"]}
      )

