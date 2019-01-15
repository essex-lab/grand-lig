"""
Setup script to facilitate installation of the GCMC OpenMM scripts

Marley L. Samways
"""

#from distutils.core import setup
from setuptools import setup

setup(name="grand",
      version="0.1.0",
      description="OpenMM-based implementation of grand canonical Monte Carlo (GCMC)",
      author="Marley L. Samways",
      author_email="mls2g13@soton.ac.uk",
      packages=["grand", "grand.tests"],
      setup_requires=["pytest-runner"],
      tests_require=["pytest"],
      test_suite="grand.tests",
      package_data={"grand": ["data/*"]}
      )

