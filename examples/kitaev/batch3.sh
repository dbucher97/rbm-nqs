#! /usr/bin/env bash

export DYLD_LIBRARY_PATH=$HOME/boost-gcc/lib:
export OMP_NUM_THREADS=1

rm n3_bs*

n=50

st=9

for x in $(seq 0 $n); do
k=$(($RANDOM % $st))
echo $x
for i in {0..9} 95 975 99999999999999999; do
  mpirun -n 4 rbm -i kitaev3.ini --evaluate --sampler.metropolis.bond_flips=0.$i \
  --seed=$RANDOM --rbm.file.name="st_$k.state" >> "n3_bs$x.plog"
done
done
