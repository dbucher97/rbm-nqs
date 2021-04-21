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

namespace model {
// Definition of the 2 site Pauli matrices.
Eigen::Matrix4cd sxsx =
    ((Eigen::Matrix4cd() << 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0)
         .finished());
Eigen::Matrix4cd sysy =
    ((Eigen::Matrix4cd() << 0, 0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, 0)
         .finished());
Eigen::Matrix4cd szsz =
    ((Eigen::Matrix4cd() << 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1)
         .finished());
}  // namespace model

using namespace model;

kitaev::kitaev(size_t size, const std::array<double, 3>& J) {
    lattice_ = std::make_unique<lattice::honeycomb>(size);
    hamiltonian_ = std::make_unique<operators::bond_op>(
        lattice_->get_bonds(),
        std::vector<Eigen::MatrixXcd>{J[0] * sxsx, J[1] * sysy, J[2] * szsz});
}
