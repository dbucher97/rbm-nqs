# RBM-NQS Application

This repository features a Restricted Boltzmann Machine (RBM) implementation
for Kitaev's toric code and the Kitaev Honeycomb Model (KHM). The code is easily
extensible with new models and lattices.

### Building

The program depends on `Eigen3`, `Boost` (ProgramOptions) `MPI` and
`Googletest`, and has been only developed for UNIX based systems. The repository
comes with the Open Source Fortran Packages
[MINRES-QLP](https://www.netlib.org/toms/937) and
[Pfapack](https://arxiv.org/abs/1102.3440).

Those packages are found in the directory `deps` and need to be built and
beforehand.

Two Makefiles are listed. One is for building with `GCC` on my local machine,
the other is for building with `ICPC` on the LRZ HPC Linux cluster.
Please orient yourself upon those, when building.

Define `EIGEN_USE_MKL_ALL` for Intel `MKL` support, or `EIGEN_USE_BLAS` for
`blas` routines. Make sure `MKL` or `BLAS` is installed then.

The application should be linked into one executable, typically called `rbm`.

### Running


```
rbm --train -n [your_name] /path/to/your.ini
```

A set of example `ini` files for the Honeycomb Kitev Model can be found inside
the `params` directory.

The result of a training process will be the optimized weights of the RBM. which
are stored in `[your_name].rbm`. And the log of the training (Energy and various
other obervables depending on the kind of process chosen) is stored in
`[your_name].log`. Additionally sometimes for small system sizes one wants to
retrieve the whole quantum state (after the training). This can be achieved by
the flag `-s` or `--state`.
> This is not yet implemented however.

###  Parameters

The following `ini` file gives an overview of the available parameters with
quick explanations.

```ini
# RBM ini file with all options for the application. Default values are set.

# ======== General Parameters ==================================================
seed = 421390484             # Seed of the RNG.
deterministic = false        # Seed RNG with random device.
name = n2_a1                 # Set name of the RBM.
n_threads = 4                # Set number of OMP threads (deprecated).

n_epochs = 4000              # Number of training epochs.

seed_search = 0              # Set the number of different seeds to try.
seed_search_epochs = 1000    # Set the number of epochs after which seeds are
                             # |     compared.

# ======== Model Parameters ====================================================
[model]
type = kitaev             # Set the model type (kitaev or toric).
n_cells = 2               # Set number of unit cells (square)
n_cells_b = -1            # Set number of unit cells in the first dimension
                          # |     (if set to -1, use n_cells)
J = -1                    # Set Interaction strength.
h = 0                     # Set (potential) second interaction strength.
helper_strength = 0       # Set P_W strength.
symmetry = 0.5            # Set translational symmetry e.g. (0, 0.5, 1, 2).
lattice_type = none       # Set lattice type if special type is available
                          # |     (honeycomb: hex)

# ======== RBM Parameters ======================================================
[rbm]
type = basic              # Set RBM type (basic, symmetry, pfaffian, file).
alpha = 1                 # Set number of hidden units (as multiple of visible
                          # |     units).
weights = 1e-4            # Set stddev for weights initialization.
weights_imag = -1         # Set stddev for imag weights initialization
                          # |     (if -1 -> rbm.weights)
weights_type = none       # Initialization type for RBM weights, for special
                          # |     initial states (deprecated).
correlators = 0           # Enables correlators if set to 1 and correlators are
                          # |     available for the model (deprecated).
pop_mode = 0              # Switches between Psi calculation modes
                          # (0 = sum log cosh, 1 = prod cosh).
cosh_mode = 2             # Switches between Cosh modes (0 = std cosh + log,
                          # |     1 = approx cosh, 2 = our log cosh)
file.name = none          # Specify the filename of the state to load from.
pfaffian = false          # Enables use of pfaffian wave function addition.
[rbm.pfaffian]
symmetry = 0.5            # Set symmetry for the pfaffian parameters.
weights = 1e-2            # Set stddev of pfaffian parameters.
normalize = false         # Normalize pfaffian parameters to pfaffian prop to 1
                          # |     (deptrecated).
load = none               # Specify name of a already trained # '.rbm' of a
                          # |     pfaffian wavefunction to load.
no_updating = false       # Don't update the pfaffian context each iteration
                          # |     but calculate from scratch

# ======== Sampler Parameters ==================================================
[sampler]
type = metropolis         # Set sampler type (metropolis, full).
n_samples = 1000          # Set number of samples.
n_samples_per_chain = 0   # Set number samples per chain (overrides
                          # |     sampler.n_samples).
eval_samples = 0          # Set number of samples for evaluation.
full.n_parallel_bits = 2  # Set number of bits executed in parallel in perfect
                          # |     sampling. #MPI processes = # 2^n.
pfaffian_refresh = 0      # Set number of Xinv updates beforerecalculation from
                          # |     scratch
lut_exchange = 0          # Set number of samples before RBM LUT exchange is
                          # |     triggered (unstable).
[sampler.metropolis]
n_chains = 4              # Set number of chains in Metropolis sampling.
n_warmup_steps = 100      # Set number of warmup sweeps.
n_steps_per_sample = 1    # Set number of seeps between a sample.
bond_flips = 0.5          # Probability for bond flip for update proposal.

# ======== Optimizer Parameters ================================================
[optimizer]
type = SR                 # Set optimizer type (SR, SGD).
learning_rate = 1e-2      # Set learning rate, optionally with decay factor.

sr.reg1 = 1,1e-4,0.9      # Set regularization diagonal scaling, optionally with
                          # |     decay factor.
sr.reg2 = 1e-3,1e-6,0.9   # Set regularization diagonal shift, optionally with
                          # |     decay factor.
sr.deltareg1 = 1e-2       # Diagonal scaling offset for pfaffian parameters.
sr.method = cg            # The method for the SR solver (direct, minresqlp,
                          # |     cg, cg-direct).
sr.max_iterations = 0     # Set number of max iterations for iterative method.
sr.rtol = 0               # Set residue tolerance for the iterative method.
sgd.real_factor = 1.      # Set the factor the real part of the update vector is
                          # |     divided by.

plugin = none             # Set optional plugin for optimization (momentum,
                          # |     adam, heun).
adam.beta1 = 0.9          # Set Adam beta1.
adam.beta2 = 0.999        # Set Adam beta2.
adam.eps = 1e-8           # Set Adam eps.
mom.alpha = 0.9           # Set momentum alpha.
mom.dialup = 1.           # Set momentum dialup.
heun.eps = 1e-3           # Set heun epsilon.

resample = false          # Resample if certain conditions on nergy / variance
                          # are not fullfilled (not recommended!).
resample.alpha1 = 0.1     # Cond. 1: Energy difference smaller than alpha1.
resample.alpha2 = 5       # Cond. 2: Imaginary energy samller than alpha2 * var.
resample.alpha3 = 10      # Cond. 3: variance ratio samller than alpha3.

```

## License

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Copyright (c) 2021 David Bucher


