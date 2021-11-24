"""RBM module for C program debugging"""

import os
import numpy as np
import scipy.sparse as sp
import platform
import struct
from scipy.sparse.linalg import eigsh
from typing import Union, List, Tuple

SZ = sp.csr_matrix([[1, 0], [0, -1]])
SY = sp.csr_matrix([[0, -1j], [1j, 0]])
SX = sp.csr_matrix([[0, 1], [1, 0]])


def construct_op(bonds : List[Tuple], bond_ops : List[Tuple], N : int , J :
        List[float] = [1], log: bool = False) -> sp.csr_matrix:
    ret = sp.csr_matrix((2**N, 2**N), dtype=complex)
    c = 0
    for sites, t in bonds:
        c += 1
        ops = bond_ops[t]
        sso = list(sorted(zip(sites, ops), key=lambda x: x[0], reverse=True))
        upper = N
        x = sp.identity(1)
        for site, op in sso:
            lower = site
            x = sp.kron(x, sp.identity(2**(upper-lower-1))).tocsr()
            x.eliminate_zeros()
            upper = lower
            x = sp.kron(x, op).tocsr()
            x.eliminate_zeros()
        x = sp.kron(x, sp.identity(2**upper)).tocsr()
        x.eliminate_zeros()
        ret += J[t % len(J)] * x
        if log:
            print(f'constructed bond {c}/{len(bonds)}')
    return ret


def get_bonds_ops(model : str = "kitaev") -> List[Tuple]:
    if model == "kitaev":
        return [(SX, SX), (SY, SY), (SZ, SZ)]
    elif model == "toric":
        return [(SZ, SZ, SZ, SZ), (SX, SX, SX, SX)]
    return []

def get_hex_ops() -> List[Tuple]:
    return [(SX, SY, SZ, SX, SY, SZ)]

def get_bonds(n : int, model : str = "kitaev", args : List = []) -> Tuple[List, int]:
    prstr = ''
    if platform.system() == 'Darwin':
        prstr = 'env DYLD_LIBRARY_PATH=$HOME/boost-gcc/lib: '
    else:
        prstr = 'mpirun -n 1 '
    process = os.popen(f'{prstr}rbm --model.n_cells={n} --model.type={model} ' +
            '--print_bonds ' + ' '.join(args))

    out = process.read()
    if process.close() is not None:
        print(out)
        raise TypeError(f'"{model}" not available.')
    bonds = []
    midx = 0
    for l in out.strip().split('\n'):
        sites = l.strip().split(',')
        sites = list(map(int, sites))
        typ = sites[-1]
        del sites[-1]
        if max(sites) > midx:
            midx = max(sites)
        bonds.append((sites, typ))
    return bonds, midx + 1


def get_hamiltonian(n : int, model : str = "kitaev", log : bool = False, args :
        List = [], J : List[float] =[-1.] ) -> sp.csr_matrix:
    bonds, N = get_bonds(n, model, args)
    bond_ops = get_bonds_ops(model)

    if log:
        print(f'Building Kitaev Hamiltonian ({n})')

    return construct_op(bonds, bond_ops, N=N, J=J, log=log)

def get_hex(n : int, log : bool = False, args : List = []) -> List[sp.csr_matrix]:
    prstr = ''
    if platform.system() == 'Darwin':
        prstr = 'env DYLD_LIBRARY_PATH=$HOME/boost-gcc/lib: '
    else:
        prstr = 'mpirun -n 1 '
    process = os.popen(f'{prstr}rbm --model.n_cells={n} ' +
            '--print_hex ' + ' '.join(args))
    out = process.read()
    hex = []
    midx = 0
    for l in out.strip().split('\n'):
        s = l.strip().split(',')[:-1]
        s = list(map(int, s))
        if midx < max(s):
            midx = max(s)
        hex.append((s, 0))
    hex_ops = get_hex_ops()

    if log:
        print(f'Building Hexagon Operators ({n})')

    ret = []
    for i, h in enumerate(hex):
        ret.append(construct_op([h], hex_ops, N=midx+1, J=[1]))
        print(f'constructed hex {i+1}/{len(hex)}')
    return ret


def evaluate(state : np.ndarray, op : Union[np.ndarray, sp.csr_matrix]) -> complex:
    return state.T.conj() @ op @ state

def groundstate(ham : sp.csr_matrix) -> Tuple[float, np.ndarray]:
    v, w = eigsh(ham, k=1)
    return v[0], w[:, 0]

def load_state(name : str, N : int) -> np.ndarray:
    fmt = 'dd' * (2 ** N)
    with open(name, 'rb') as f:
        rows,  = struct.unpack('Q', f.read(8))
        cols,  = struct.unpack('Q', f.read(8))
        print(rows, cols)
        vec1 = np.array(struct.unpack(fmt, f.read(len(fmt) * 8)), dtype=np.complex128)
    vec1 = vec1.reshape(2 ** N, 2)
    vec1[:, 1] *= 1j
    vec1 = vec1.sum(axis=1)
    return vec1

def load_mat(f):
    r, c = struct.unpack('ll', f.read(16))
    nd = r * c * 2
    x = np.array(struct.unpack(f'{nd}d', f.read(8 * nd)), dtype=complex)
    x = x.reshape((-1, 2))
    x[:, 1] *= 1j
    x = np.sum(x, axis=1)
    return x.reshape(c, r).T

def load_weights(name, prefix=''):
    with open(os.path.join(prefix, name), 'rb') as f:
        weights = load_mat(f)
        v_bias = load_mat(f)
        h_bias = load_mat(f)
    return weights, v_bias, h_bias



