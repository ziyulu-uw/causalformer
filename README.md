# causalformer
 Implementation and results of Causalformer from [*Attention for Causal Relationship Discovery from Biological Neural Dynamics*](https://arxiv.org/abs/2311.06928) (accepted to the NeurIPS 2023 Workshop on Causal Representation Learning). Implementation of Causalformer is based on the [Spacetimeformer](https://arxiv.org/abs/2109.12218). Please refer to the [Spacetimeformer repo](https://github.com/QData/spacetimeformer) for dependencies of Causalformer. 

Notes:
- izhikevich_data (directory): Simulated data used in the paper. Consists of 160 different random networks in total.
  - v_alltimes.txt: Variable $v$ of each neuron at each timestep. Size: $n \times T$, $n$: number of neurons, $T$: number of timesteps.
  - u_alltimes.txt: Variable $u$ of each neuron at each timestep. Size: $n \times T$.
  - v_normed_alltimes.txt: Normalized variable $v$ of each neuron at each timestep. For each neuron, $v$ is first clipped at the firing threshold and then z-scored by its mean and std across timesteps. Size: $n \times T$.
  - inp.txt: Random thalamic input to each neuron at each timestep. Size: $n \times T$.
  - firings.txt: 1st column: timestep at which a spike is generated. 2nd column: the neuron who fired the spike at the correpsonding timestep in the 1st column. 
  - params_abcd.txt: Parameters $a, b, c, d$ for each neuron. Size: $n \times 4$.
  - connectivity.txt: Connectivity matrix $M$. $M(i, j) > 0$: excitatory connection from neuron $j$ to $i$. $M(i, j) < 0$: inhibitory connection from neuron $j$ to $i$. $M(i, j) = 0$: no connection from neuron $j$ to $i$. Magnitude of $M(i, j)$ indicates connectivity strength. Size: $n \times n$.
  - mvgc_{z}_F_p{x}.txt: F-test statistic for time-domain pairwise conditional Granger causalities, returned by the MVGC routine. {x}: MVGC maximum model orders. {z}: 'aic' or 'bic', model selection criterion.
- run_izhikevich_sim.m: Matlab code for data simulation using the [Izhikevich model](https://www.izhikevich.org/publications/spikes.htm) and running [Multivariate Granger Causality (MVGC)](https://www.mathworks.com/matlabcentral/fileexchange/78727-the-multivariate-granger-causality-mvgc-toolbox) analysis.
- causalformer_model (directory): Pytorch implementation of Causalformer.
- randnet_results (directory): Summary of experiment results on the data from izhikevich_data.
- plot_randnet_results.ipynb: Notebook for visualizing results in randnet_results. Reproduces figure 2, figure 4(b) in the paper. 
- example_trained_models (directory): Trained Causalformer models on one example random network from izhikevich_data (5 neurons, connection probablity 0.4, random topology 4). Consists of 10 models with different random initializations (i.e., random seeds).
- results_visualization.ipynb: Notebook for visualizing results in example_trained_models. Reproduces figure 1(a)(b)(d)(e), figure 4(a) in the paper. 
