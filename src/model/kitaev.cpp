/*
 * Copyright (c) 2021 David Bucher <David.Bucher@physik.lmu.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include <Eigen/Dense>
//
#include <model/kitaev.hpp>
#include <operators/local_op.hpp>
#include <operators/local_op_chain.hpp>

using namespace model;

kitaev::kitaev(size_t size, const std::array<double, 3>& J, int size_b,
               bool full_symm) {
    lattice_ = std::make_unique<lattice::honeycomb>(size, size_b, full_symm);
    std::vector<SparseXcd> bond_ops = {
        J[0] * kron({sx(), sx()}),
        J[1] * kron({sy(), sy()}),
        J[2] * kron({sz(), sz()}),
    };
    hamiltonian_ =
        std::make_unique<operators::bond_op>(lattice_->get_bonds(), bond_ops);
}

void kitaev::add_helper_hamiltonian(double strength) {
    auto hex =
        dynamic_cast<lattice::honeycomb*>(lattice_.get())->get_hexagons();
    SparseXcd plaq_op = kron({sx(), sy(), sz(), sx(), sy(), sz()});
    SparseXcd idn(plaq_op.rows(), plaq_op.cols());
    idn.setIdentity();
    plaq_op -= idn;
    plaq_op *= -strength;
    helpers_ = hex.size();
    for (auto& h : hex) {
        hamiltonian_->push_back(operators::local_op(h, plaq_op));
    }
}
