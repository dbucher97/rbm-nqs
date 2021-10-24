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
#include <unsupported/Eigen/KroneckerProduct>
//
#include <lattice/toric_lattice.hpp>
#include <model/toric.hpp>
#include <operators/local_op.hpp>
#include <operators/local_op_chain.hpp>

using namespace model;

toric::toric(size_t size, double J = -1.)
    : plaq_{J * kron({sz(), sz(), sz(), sz()})},
      vertex_{J * kron({sx(), sx(), sx(), sx()})} {
    auto lat = new lattice::toric_lattice(size);
    auto ham = new operators::local_op_chain();
    auto plaqs = lat->construct_plaqs();
    for (auto& p : plaqs) {
        std::vector<size_t> v(p.idxs.begin(), p.idxs.end());
        if (p.type == 0) {
            ham->push_back(operators::local_op(v, plaq_));
        } else {
            ham->push_back(operators::local_op(v, vertex_));
        }
    }
    lattice_ = std::unique_ptr<lattice::bravais>(lat);
    hamiltonian_ = std::unique_ptr<operators::local_op_chain>(ham);
}
