# Kitaev Honeycomb Model with Restricted Boltzmann Machine

This repository features a Restricted Boltzmann Machine (RBM) implementation
for the Kitaev Honeycomb Model. The code is easily extensible with new models
and lattices. However, by now, it solely focuses on the Kitaev Honeycomb Model.

The RBM consists of a number of weights connecting the hidden nodes with the
visible nodes. Each of the visible nodes is a spin and can have the value -1 or
1 corresponding to the measurement results in the $`z`$-basis.

### Building

The program is dependend of `Eigen3` and `Boost` and is only developed for UNIX
based systems. To build the code, simply run

```
make
```

The documentation is built upon `Doxygen` and can be built by running

```
make doc
```

### Training the RBM

### Theoretical Backgorund

```math
| \psi\rangle = \sum\_{\{\sigma\}}\psi{\sigma}|\sigma\rangle
```
