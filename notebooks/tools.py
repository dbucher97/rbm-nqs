#            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
#                    Version 2, December 2004
#
# Copyright (C) 2021 David Bucher <David.Bucher@physik.lmu.de>
#
# Everyone is permitted to copy and distribute verbatim or modified
# copies of this license document, and changing it is allowed as long
# as the name is changed.
#
#            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
#   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION
#
#  0. You just DO WHAT THE FUCK YOU WANT TO.

#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse import linalg

import numba

# Pauli Operators
s_x = np.array([[0, 1], [1, 0]])
s_y = np.array([[0, -1j], [1j, 0]])
s_z = np.array([[1, 0], [0, -1]])

N = 18


def load_bonds(filename):
    bonds = []
    with open(filename, 'r') as f:
        for line in f:
            a, b, typ = map(int, line.split(','))
            bonds += [(a, b, typ)]
    return bonds

def get_U():
    Ux = np.array([[1, 1], [1, -1]])/np.sqrt(2)
    Uy = np.array([[1, -1j], [1, 1j]])/np.sqrt(2)
    Uz = np.eye(2)
    return Ux, Uy, Uz

def get_bond_ops_S3():
    Ux, Uy, Uz = get_U()

    sx_y = Uy @ s_x @ Uy.T.conj()
    sx_z = s_x

    sy_x = Ux @ s_y @ Ux.T.conj()
    sy_z = s_y

    sz_x = Ux @ s_z @ Ux.T.conj()
    sz_y = Uy @ s_z @ Uy.T.conj()

    x_yz = (sx_y, sx_z)
    x_zy = (sx_z, sx_y)
    y_xz = (sy_x, sy_z)
    y_zx = (sy_z, sy_x)
    z_xy = (sz_x, sz_y)
    z_yx = (sz_y, sz_x)

    zz = (s_z, s_z)

    return [zz, zz, zz, x_yz, x_zy, y_xz, y_zx, z_xy, z_yx]


def get_bond_ops():
    return [(s_x, s_x), (s_y, s_y), (s_z, s_z)]


def build_op(bonds, bond_ops, N=18, J=[-1]):
    op = sparse.csr_matrix((2**N, 2**N), dtype=complex)
    for a, b, t in bonds:
        opa, opb = bond_ops[t]
        if a > b:
            a, b = b, a
            opa, opb = opb, opa
        x = sparse.kron(sparse.identity(2**(N-b-1)), opb)
        x = sparse.kron(x, sparse.identity(2**(b-a-1)))
        x = sparse.kron(x, opa)
        x = sparse.kron(x, sparse.identity(2**a))
        op += J[t % len(J)] * x
    return op

def build_op2(bonds, bond_ops, N=18, J=[-1]):
    ret = sparse.csr_matrix((2**N, 2**N), dtype=complex)
    print(N)
    for sites, t in bonds:
        ops = bond_ops[t]
        sso = list(sorted(zip(sites, ops), key=lambda x: x[0], reverse=True))
        upper = N
        x = sparse.identity(1)
        for site, op in sso:
            lower = site
            x = sparse.kron(x, sparse.identity(2**(upper-lower-1)))
            upper = lower
            x = sparse.kron(x, op)
        x = sparse.kron(x, sparse.identity(2**upper))
        ret += J[t % len(J)] * x
    return ret

def plot_state(*states):
    plt.plot(range(len(states[0])), *[np.abs(state) for state in states])


def solve_op(op, k=6):
    w, v = linalg.eigs(op, k=k)
    print(w)
    e0 = min(w)
    v = v[:, np.abs(w - e0) < 1e-10].T
    return e0, v


def overlap(state1, state2):
    return np.abs(state1.conj().dot(state2))**2


def total_overlap(state, v):
    ret = 0
    for vi in v:
        ret += overlap(state, vi)
    return ret


def store_state(state, filename):
    with open(filename, 'w') as f:
        for x in state:
            f.write(f'({np.real(x)},{np.imag(x)})\n')


@numba.jit
def num_to_vec(num, N=18):
    l = np.zeros((N))
    for i in range(N):
        l[i] = -1 if (num >> i) & 1 else 1
    return l


@numba.jit
def vec_to_num(vec, N=18):
    r = 0
    for i in range(N):
        if(vec[i] < 0):
            r += 1 << i
    return r


@numba.jit
def vec_to_num_loc(vec, acts_on):
    r = 0
    for i, a in enumerate(acts_on):
        if(vec[a] < 0):
            r += 1 << i
    return r


@numba.jit
def vec_change_num_loc(vec, acts_on, num):
    v = np.copy(vec)
    for i, a in enumerate(acts_on):
        v[a] = -1 if (num >> i) & 1 else 1
    return v


def calc_bond(state, bond, bond_ops):
    acts_on = np.array((bond[0], bond[1]))
    oa, ob = bond_ops[bond[2]]
    op = -np.kron(ob, oa)
    res = 0
    for num, psi in enumerate(state):
        vec = num_to_vec(num)
        n2 = vec_to_num_loc(vec, acts_on)
        v2 = np.array([0, 0, 0, 0])
        v2[n2] = 1
        op2 = np.dot(v2, op)
        for i in range(4):
            if(np.abs(op2[i]) > 1e-12):
                psi2 = state[vec_to_num(vec_change_num_loc(vec, acts_on, i))]
                # print(np.round(psi, 3), np.round(psi2, 3), op2[i], i == n2, vec_change_num_loc(vec, acts_on, i))
                res += psi2 / psi * op2[i] * np.abs(psi)**2
    return res


def load_state(path):
    sim_state = []
    with open(path, 'r') as f:
        for l in f:
            r, i = l.strip().strip('()').split(',')
            sim_state.append(float(r) + 1j * float(i))
    sim_state = np.array(sim_state)
    norm = np.linalg.norm(sim_state)
    return sim_state / norm, norm


def energy(op, state):
    return state.conj().dot(op.dot(state)) / np.log2(len(state))

def calc(op, state):
    return state.conj().dot(op.dot(state))
