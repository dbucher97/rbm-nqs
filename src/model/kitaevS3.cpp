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
#include <complex>
//

#include <model/kitaevS3.hpp>

model::kitaevS3::kitaevS3(size_t size, const std::array<double, 3>& J) {
    if (size % 3 != 0)
        throw std::runtime_error("Size not valid for Kitaev S3.");
    lattice_ = std::make_unique<lattice::honeycombS3>(size / 3);
    std::vector<const SparseXcd> bond_ops = {
        J[0] * kron({sz(), sz()}),
        J[1] * kron({sz(), sz()}),
        J[2] * kron({sz(), sz()}),
        J[0] * kron({sy(), sx()}),
        J[0] * kron({sx(), sy()}),
        -J[1] * kron({sy(), sy()}),
        -J[1] * kron({sy(), sy()}),
        J[2] * kron({sx(), sx()}),
        J[2] * kron({sx(), sx()})
    };
    hamiltonian_ = std::make_unique<operators::bond_op>(
        lattice_->get_bonds(),
        bond_ops
    );
}
