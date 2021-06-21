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
#include <unsupported/Eigen/KroneckerProduct>
//

#include <model/isingS3.hpp>

model::isingS3::isingS3(size_t size, double J) {
    if (size % 3 != 0)
        throw std::runtime_error("Size not valid for Kitaev S3.");
    lattice_ = std::make_unique<lattice::honeycombS3>(size / 3);
    std::vector<SparseXcd> bond_ops = {
        J * kron({sx(), sx()}),
        J * kron({sx(), sx()}),
        J * kron({sz(), sz()}),
        J * kron({sx(), sz()}),
        J * kron({sz(), sx()}),
        J * kron({sx(), sz()}),
        J * kron({sz(), sx()}),
        J * kron({sx(), sx()}),
        J * kron({sx(), sx()})
    };
    hamiltonian_ = std::make_unique<operators::bond_op>(
        lattice_->get_bonds(),
        bond_ops);
        /* std::vector<Eigen::MatrixXcd>{J * ixx, J * iyy, J * izz, J * ix_yz,
                                      J * ix_zy, J * iy_zx, J * iy_xz,
                                      J * iz_xy, J * iz_yx}); */
}
