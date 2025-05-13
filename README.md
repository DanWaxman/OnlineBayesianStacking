# Online Bayesian Stacking (OBS)

This repository contains the code for the experiments presented in the paper "Bayesian Ensembling: Insights from Online Optimization and Empirical Bayes". The project introduces Online Bayesian Stacking (OBS), a method for adaptively combining Bayesian models in an online/sequential learning setting, drawing connections to Online Portfolio Selection (OPS) and Online Convex Optimization (OCO).

The code demonstrates the application of OBS to various scenarios and compares its performance against Online Bayesian Model Averaging (O-BMA).

## Directory Structure

The project is organized into several experiment-specific directories, each containing its own README:

* `OnlineForecastingExperiment/`: Code for online forecasting of S&P 500 data using ensembles of GARCH models, estimated with Sequential Monte Carlo (SMC).
* `OnlineGPsExperiment/`: Code for online Gaussian Process (GP) regression in non-stationary environments, using ensembles of Dynamic Online Ensemble Basis Expansions (DOEBE) models.
* `OnlineVIExperiment/`: Code for online classification on MNIST using ensembles of Bayesian Neural Networks learned with Online Variational Inference (specifically, BONG).
* `ToyExampleExperiment/`: Code for the subset linear regression toy example designed to illustrate differences between OBS and BMA in M-open and M-closed settings.

## Core Idea

The project explores Online Bayesian Stacking (OBS) as an alternative to Online Bayesian Model Averaging (O-BMA). OBS frames the problem of finding optimal ensemble weights as an online portfolio selection problem, allowing the use of efficient algorithms from online convex optimization like:
* Exponentiated Gradients (EG)
* Online Newton Step (ONS)
* Soft-Bayes

These methods are used to adaptively update the weights of component Bayesian models based on their sequential predictive performance (log-likelihood).

## Running the Experiments

Each experiment directory (`OnlineForecastingExperiment`, `OnlineGPsExperiment`, `OnlineVIExperiment`, `ToyExampleExperiment`) contains its own `README` or main script with instructions on how to run the respective experiments. Generally, the workflow involves:
1.  Running a script (e.g., `MAIN_diff_garch_models.py`, `main_save_pll_models.py`, `mnist_clf.py`, `subset_linear_regression.py`) to generate or load data, train/run individual models (or pre-trained model PLLs), and save the per-model predictive log-likelihoods over time.
2.  For some experiments (like `OnlineGPsExperiment`), a script like `compute_obs_weights.py` is then used to apply the OBS/O-BMA algorithms to these saved PLLs.
3.  Plotting scripts or notebooks (e.g., `plot_obs_weights.py` or an implied `results.ipynb`) are used to visualize the results.

Please refer to the `README` files within each specific experiment directory for detailed instructions.

## Dependencies

Because each experiments depends on differing (often conflicting) libraries, the specific dependencies are different for each experiment.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
Copyright (c) 2025 Daniel Waxman
