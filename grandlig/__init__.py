"""
grandlig

OpenMM-based implementation of grandlig canonical Monte Carlo (GCMC) moves to sample ligand positions

William G. Poole
Marley L. Samways
Ollie Melling
"""

__version__ = "1.0.0"

from grandlig import samplers, utils, potential, tests
# from ._version import __version__

print(
    "Hi. This is the current implementation of GCMC/GCNCMC for small \
       molecule ligands. While this will work"
    "with water molecules as well, the original implementation can be \
              found at: https://github.com/essex-lab/grand"
)

print("If you are only interested in water simulations, we recommend using the original implementation while this module undergoes more extensive testing.")

"""GCMC Sampling"""

