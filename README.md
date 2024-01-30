# Code for Approximating the Shapley Value without Marginal Contributions
This folder contains the code to run the experiments conducted for the submission "Approximating the Shapley Value without Marginal  Contributions".

# How to run the experiments

There exists a specific file for each game.
For example, in order to run the Airport game, execute the file "run_airport.py".
The results are printed into a file into the Data folder for each algorithm used.

You can outcomment algorithms that you do not wish to run.

For the algorithms contained in the list "approx_methods", a .txt file is created, and for those in "approx_methods_extra" a .csv file is created.
Both files contain the columns t (the current time step), mse (the mean squared error average over all players for that time step), and mseStdErr (the estimated standard variance of the MSE values at that time over all repetitions).

The results are averaged over all repetitions (NUMBER_OF_RUNS).

# Contained Algorithms
- ApproShapley
- Stratifified Sampling
- Structured Sampling
- Owen Sampling
- KernelSHAP
- Unbiased KernelSHAP
- SVARM
- Stratified SVARM
- Stratified SVARM+

# Contained Games
- Airport game
- SOUG Game
- Shoe game
- NLP sentiment game
- Image classifier game
- Adult classification game