Example 1: T4 Lysozyme Site Hunting
========

In this first example we are going to use grandlig as a "site-finding tool" where by we assume no prior knowledge of the system.

T4L99A is an engineered mutant with a small, occuluded binding pocket known to bind small aromatic molecules such as benzene.

We are going to perform a GCNCMC simulation with the GCMC region set to cover the whole protein.

To run this script with the provided input files in ``grandlig/examples/lysozyme_bindingsite``:

.. code-block:: bash

    python Benzene_Binding_site.py -p T4L99A_equiled.pdb -l Benzene.pdb -x Benzene.xml -c 0.5


.. literalinclude:: ../../examples/lysozyme_bindingsite/Benzene_Binding_site.py
    :language: python


   