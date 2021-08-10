"""RBM module for C program debugging"""

import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from typing import Union

SZ = sp.csr_matrix(np.array([[1, 0], [0, -1]]))
SY = sp.csr_matrix(np.array([[0, -1j], [1j, 0]]))
SX = sp.csr_matrix(np.array([[0, 1], [1, 0]]))


def construct_op(bonds : list[tuple], bond_ops : list[tuple], N : int , J : list[int] = [1]) -> sp.csr_matrix:
    ret = sp.csr_matrix((2**N, 2**N), dtype=complex)
    for sites, t in bonds:
        ops = bond_ops[t]
        sso = list(sorted(zip(sites, ops), key=lambda x: x[0], reverse=True))
        upper = N
        x = sp.identity(1)
        for site, op in sso:
            lower = site
            x = sp.kron(x, sp.identity(2**(upper-lower-1)))
            upper = lower
            x = sp.kron(x, op)
        x = sp.kron(x, sp.identity(2**upper))
        ret += J[t % len(J)] * x
    return ret


def get_bonds_ops(model : str = "kitaev") -> list[tuple]:
    if model == "kitaev":
        return [(SX, SX), (SY, SY), (SZ, SZ)]
    return []


def get_hamiltonian(n : int, model : str = "kitaev") -> sp.csr_matrix:
    process = os.popen(f'env DYLD_LIBRARY_PATH=/Users/david/boost-gcc/lib: rbm --model.n_cells={n} --model.type={model} --print_bonds')
    out = process.read()
    if process.close() is not None:
        raise TypeError(f'"{model}" not available.')
    bonds = []
    midx = 0
    for l in out.strip().split('\n'):
        a, b, typ = l.strip().split(',')
        a = int(a)
        b = int(b)
        if a > midx:
            midx = a
        if b > midx:
            midx = b
        bonds.append(([a, b], int(typ)))
    bond_ops = get_bonds_ops(model)

    return construct_op(bonds, bond_ops, N=midx+1, J=[-1])

def evaluate(state : np.ndarray, op : Union[np.ndarray, sp.csr_matrix]) -> complex:
    return state.T.conj() @ op @ state

def groundstate(ham : sp.csr_matrix) -> tuple[float, np.ndarray]:
    v, w = eigsh(ham, k=1)
    return v[0], w[:, 0]
