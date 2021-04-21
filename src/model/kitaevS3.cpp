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

namespace model {
// Definition of the 2 site Pauli matrices.
Eigen::Matrix4cd zz =
    ((Eigen::Matrix4cd() << 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1)
         .finished());
Eigen::Matrix4cd x_zy =
    std::complex<double>(0, 1.) *
    ((Eigen::Matrix4cd() << 0, 0, 0, -1, 0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0)
         .finished());
Eigen::Matrix4cd x_yz =
    std::complex<double>(0, 1.) *
    ((Eigen::Matrix4cd() << 0, 0, 0, -1, 0, 0, 1, 0, 0, -1, 0, 0, 1, 0, 0, 0)
         .finished());
Eigen::Matrix4cd y_xz =
    ((Eigen::Matrix4cd() << 0, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 0, 1, 0, 0, 0)
         .finished());
Eigen::Matrix4cd y_zx = y_xz;
Eigen::Matrix4cd z_xy =
    ((Eigen::Matrix4cd() << 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0)
         .finished());
Eigen::Matrix4cd z_yx = z_xy;
}  // namespace model

model::kitaevS3::kitaevS3(size_t size, const std::array<double, 3>& J) {
    if (size % 3 != 0)
        throw std::runtime_error("Size not valid for Kitaev S3.");
    lattice_ = std::make_unique<lattice::honeycombS3>(size / 3);
    hamiltonian_ = std::make_unique<operators::bond_op>(
        lattice_->get_bonds(),
        std::vector<Eigen::MatrixXcd>{J[0] * zz, J[1] * zz, J[2] * zz,
                                      J[0] * x_yz, J[0] * x_zy, J[1] * y_xz,
                                      J[1] * y_zx, J[2] * z_xy, J[2] * z_yx});
}
