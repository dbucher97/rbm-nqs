#! /usr/bin/env bash

export DYLD_LIBRARY_PATH=$HOME/boost-gcc/lib:
export OMP_NUM_THREADS=1

function run() {
  name=$1
  shift
  mpirun -n 4 rbm -i toric.ini --name="$name" --train -f $@ | tee "$name.plog"
  if [ "$name" == toric_n2 ]; then
    mpirun -n 4 rbm -i toric.ini --name="$name" --evaluate | tee -a "$name.plog"
  else
    mpirun -n 4 rbm -i toric.ini --name="$name" --evaluate $@ | tee -a "$name.plog"
  fi
}

echo "===== N=2 Perfect Sampling ====="
name=toric_n2
run $name --sampler.type=full

echo -e "\n\n===== N=3 Metropolis Sampling ====="
name=toric_n3
run $name --model.n_cells=3 --seed=342539432

echo -e "\n\n===== N=3 Metropolis Sampling no plaq ====="
name=toric_n3_p0
run $name --model.n_cells=3 --sampler.metropolis.bond_flips=0 --seed=342539434

echo -e "\n\n===== N=4 Metropolis Sampling ====="
name=toric_n4
run $name --model.n_cells=4 --seed=34253943324

echo -e "\n\n===== N=6 Metropolis Sampling ====="
name=toric_n6
run $name --model.n_cells=6 --seed=3425395312

echo -e "\n\n===== N=8 Metropolis Sampling ====="
name=toric_n8
run $name --model.n_cells=8 --seed=342539324213
