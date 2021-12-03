# Kitaev Honeycomb Model with Restricted Boltzmann Machine

This repository features a Restricted Boltzmann Machine (RBM) implementation
for the Kitaev Honeycomb Model. The code is easily extensible with new models
and lattices. However, by now, it solely focuses on the Kitaev Honeycomb Model.

The RBM consists of a number of weights connecting the hidden nodes with the
visible nodes. Each of the visible nodes is a spin and can have the value -1 or
1 corresponding to the measurement results in the $`z`$-basis.

### Building

The program is dependend of `Eigen3` and `Boost` and is only developed for UNIX
based systems. To build the code, simply run `make`

The `rbm` base executable is located in the `build` directory.


### Running

The Program options is based on the Boost program options. You can get an
overview of the available options with the `--help` flag. I would recommend
using an `ini` file for defining the options. For training an RBM with specific
options run. Also I would recommend setting a specific name for a simulation
with the `name` option or `-n/--name` command line options.

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

## Explanation

This section guides through the workings of the RBM program and explaines all
the parameters on the go.

### General settings

The following options are available

| Option | Explanation |
| ------ | ----------- |
| `--seed [some_seed]` | Sets the seed for the `MT19937` random number generator |
| `-h/--help` | Prints the available parameters |
| `-n/--name [your_name]` | Sets the name of the current simulation and the filename of the weights and log, respectively |
| `-t/--n_threads [threds]` | Sets the number of threads used for OpenMP |
| `-i/--infile` | loads a ini file for parameters |
| `--n_epochs [epochs]` | sets the number of epochs to train |

```ini
seed = 3219872350     # Seed

[model]
type = kitaev
J = -1
helper_strength = 0
symmetry = 0.5

[rbm]
type = basic  # symmetry, pfaffian, file
weights = 1e-4
alpha = 1

[sampler]
type = metropolis # full
n_samples = 1000

[sampler.metropolis]
n_steps_per_sample = 1
n_warmup_steps = 100
bond_flips = 0.5

[optimizer]
type = SR
learing_rate = 1e-2

plugin = # momentum, adam, heun

sr.reg1 = 10,1e-4,0.9

```

