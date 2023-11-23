# causalformer
 Implementation and results of Causalformer from [*Attention for Causal Relationship Discovery from Biological Neural Dynamics*](https://arxiv.org/abs/2311.06928) (accepted to the NeurIPS 2023 Workshop on Causal Representation Learning). Implementation of Causalformer is based on the [Spacetimeformer](https://arxiv.org/abs/2109.12218). Please refer to the [Spacetimeformer repo](https://github.com/QData/spacetimeformer) for dependencies of Causalformer. 

Notes:
- izhikevich_data (directory): Simulated data used in the paper. Consists of 160 different random networks in total.
- run_izhikevich_sim.m: Matlab code for data simulation using the [Izhikevich model](https://www.izhikevich.org/publications/spikes.htm) and running [Multivariate Granger Causality (MVGC)](https://www.mathworks.com/matlabcentral/fileexchange/78727-the-multivariate-granger-causality-mvgc-toolbox) analysis.
- causalformer_model (directory): Pytorch implementation of Causalformer.
- randnet_results (directory): Summary of experiment results on the data from izhikevich_data.
- plot_randnet_results.ipynb: Notebook for visualizing results in randnet_results. Reproduces figure 2, figure 4(b) in the paper. 
- example_trained_models (directory): Trained Causalformer models on one example random network from izhikevich_data (5 neurons, connection probablity 0.4, random topology 4). Consists of 10 models with different random initializations (i.e., random seeds).
- results_visualization.ipynb: Notebook for visualizing results in example_trained_models. Reproduces figure 1(a)(b)(d)(e), figure 4(a) in the paper. 
