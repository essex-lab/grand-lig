# Dilute benzene parallel

Here we calculate the infinitley dilute excess chemical potential of benzene using parallelisation.
```
python calcMuEx_individual.py -p benzene_water.pdb -x Benzene.xml -lam $LAMBDA
```
where LAMBDA takes the value 0 to 19. 

You should run this script in a seperate folder for each value of lambda. The resulting output is a U_ijk matrix for each lambda. (n_lambdas, n_lambda, n_samples). 

The final free energy can then be calculated using the following from the folder above the individual lambda folders:
```
python part_2_analyse_muex.py
```

This script can be adapted for multiple repeats. 
