#!/usr/bin/env sh

LOAD=load
COMPILER=gcc

while [ ! -z $1 ]; do
    if [ $1 == "-u" ]; then
        LOAD=unload
    else
        COMPILER=$1
    fi
    shift
done

if [ $COMPILER == "gcc" ]; then
    module $LOAD gcc/10.2.0
    module $LOAD boost/1.75.0-gcc8
    module $LOAD intel-mkl/2020-gcc8
else
    echo Compiler $COMPILER not available!
fi
