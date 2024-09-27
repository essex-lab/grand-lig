Example 2: Titration Overview
========
To run a GCNCMC titration you need to peform simulations of the same system at multiple B values. The script provided in ``grandlig/examples/Basic_Simulation`` provides a convienient way to 
run multiple simulations at different B values by changing the --B argument. 

To run this script for T4L99A and Benzene:

.. code-block:: bash

    python basic_NCMC_sim_B.py --pdb Native_APO.pdb --lig Benzene_H.pdb --xml Benzene_H.xml --B B_VALUE --st 150 --sph_resns LEU ALA --sph_resis 85 100 --rad 8 --hmr --nmoves 1500 --mdpermove 500


The following B values should give a good titration:
[-21.31, -19.01, -16.71, -14.4, -13.7, -12.99, -12.28, -11.57, -10.86, -10.15, -9.44, -8.74, -8.03, -7.32, -6.61, -5.9, -5.19, -2.89, -0.59, 1.71]

You can then calculate the free energy of transfer from gas phase to complex but finding the B value where the occupancy is 50% (B50). 



.. literalinclude:: ../../examples/Basic_Simulation/basic_NCMC_sim_B.py
    :language: python

Analysis can be performed using the following script. Note that occupancies are hard coded in for four repeats as an example. The script will need to be adapted to your own use case.

.. literalinclude:: ../../examples/lysozyme_titration/Plot_Titration_curve_one_lig_B_values_V2.py
    :language: python
   