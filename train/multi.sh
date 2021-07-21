#! /usr/bin/env sh

while true; do
  seed=$RANDOM
  env DYLD_LIBRARY_PATH=$HOME/boost-gcc/lib rbm --train -f -i n2.ini --name="_n3_$seed" --seed=$seed
done
