# Folder for MuEx calculations

A pre-requisite for GCNCMC calculations is the excess chemical potential (mu_ex) of the molecule being simualated. This can be calculated by performing a basic hydration free energy calculation of your molecule. 

We provide a useful function for this, but you may also use your own protocols. Note however, best practises would encourage using the same software, MD parameters and forcefields as the production GCMC simulations.

As will be discussed in the paper, a hydration free energy calculation is traditionally performed at "infinite dilution" e.g. 1 molecule in a box of water. This is an approximation and usually holds quite well when the intended concentration is low, such as those at which molecules bind.

However, at higher concentrations, the approximation breaks down because interactions with other molecules of the same type take effect. Imagine the free energy of adding a Benzene molecule to a water box vs. a 0.5 M solution of water+benzene. It would be a lot more favourable in the later. 

This is an example with Benzene at 0.5M and Toluene at infinite dilution.

