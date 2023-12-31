# Stellar Modelling Project

Run `python stellar_model.py` to run all of the code.  You can edit `config.yaml` to tune it to the star of your choice.  Note that the energy generation is tuned to a $2 M_{\odot}$ star via figure 18.7 of Kippenhahn and Wigner, and will need to be hand-tuned to fit for another mass.
`stellar_utils.py` contains all of the stellar modelling calculations, and `shooting_fn.py` contains the functions for fitting boundary conditions.  `constants.py` was provided by Prof. Schlaufman.  Utilities to plot and save individual runs are stored in `stellar_model.py`, if it is imported.

A csv of the final star is stored in `star_result.csv`.