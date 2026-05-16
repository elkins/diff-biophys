# 🚀 Use Cases

DiffBiophys enables a wide range of applications by bridging the gap between structural models and experimental solution-state data through differentiability.

---

## 1. Experimental Structure Refinement
Traditional structure refinement often relies on stochastic sampling (e.g., Monte Carlo or Simulated Annealing). With DiffBiophys, you can use **gradient-based optimization** to refine a structure directly against experimental data.

*   **Example**: Start with a homology model and refine the backbone $\phi/\psi$ angles to minimize the difference between calculated and experimental Residual Dipolar Couplings (RDCs).
*   **Key Advantage**: Faster convergence and the ability to handle high-dimensional parameter spaces.

## 2. Physics-Informed Machine Learning
Integrate biophysical "truth" into your AI models. DiffBiophys kernels can be used as **differentiable loss functions** in deep learning pipelines (e.g., training a model to predict protein conformations).

*   **Example**: Use the `debye_saxs` kernel as a loss term in a Variational Autoencoder (VAE) to ensure the latent space represents physically plausible, compact structures that match experimental scattering curves.
*   **Key Advantage**: Forces the model to learn physics-consistent representations without requiring massive labeled datasets.

## 3. Ensemble Weight Optimization
Experimental data like SAXS often represent an average over a conformational ensemble. DiffBiophys allows you to differentiate with respect to **ensemble weights**.

*   **Example**: Given a library of 100 possible protein conformations, optimize the population weights $\{w_i\}$ such that the ensemble-averaged SAXS curve $\sum w_i I_i(q)$ matches experimental data.
*   **Key Advantage**: Native JAX `vmap` support makes calculating intensities for large ensembles extremely efficient on GPUs.

## 4. Differentiable Molecular Dynamics Analysis
Use biophysical gradients to steer or analyze MD trajectories.

*   **Example**: Calculate the "experimental force" acting on a structure by taking the gradient of an experimental misfit (e.g., $\nabla_\text{coords} \|I_\text{calc} - I_\text{exp}\|^2$).
*   **Key Advantage**: Directly couples structural dynamics to solution-state observables.

## 5. Automated Tensor Fitting
In NMR, the alignment tensor (Saupe tensor) is often an unknown parameter. DiffBiophys provides built-in SVD fitting that is fully differentiable.

*   **Example**: Simultaneously solve for the optimal alignment tensor and the optimal structure by alternating between tensor fitting and coordinate optimization.
*   **Key Advantage**: Eliminates the need for manual, iterative tensor estimation.
