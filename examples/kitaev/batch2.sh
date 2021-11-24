#! /usr/bin/env bash

function run() {
  name=$1
  shift
  mpirun -n 4 rbm -i kitaev2.ini --name="${name}_pw" --train -f --evaluate \
  --model.helper_strength=0.2 --seed=$RANDOM $@ | tee "${name}_pw.plog"
  # mpirun -n 4 rbm -i kitaev2.ini --name="$name" --train -f --evaluate $@ \
  #   --seed=$RANDOM | tee "$name.plog"
}


export DYLD_LIBRARY_PATH=$HOME/boost-gcc/lib:
export OMP_NUM_THREADS=1

echo "== No symm =="
run n2_basic --rbm.type=basic --rbm.alpha=2

echo "== uc symm =="
run n2_s1 --model.symmetry=1 --rbm.alpha=3

echo "== full symm =="
run n2_s05 --model.symmetry=0.5
