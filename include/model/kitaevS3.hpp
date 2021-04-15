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

#pragma once

#include <array>
#include <complex>
#include <stdexcept>
#include <vector>
//
#include <lattice/honeycombS3.hpp>
#include <model/kitaev.hpp>
#include <operators/base_op.hpp>
#include <operators/bond_op.hpp>

/**
 * @brief Namespace for the physical models.
 */
namespace model {

extern Eigen::Matrix4cd zz;    ///< 2 site Pauli-z operator
extern Eigen::Matrix4cd x_yz;  ///< 2 site Pauli-x operator in y z basis
extern Eigen::Matrix4cd x_zy;  ///< 2 site Pauli-x operator in z y basis
extern Eigen::Matrix4cd y_xz;  ///< 2 site Pauli-y operator in x z basis
extern Eigen::Matrix4cd y_zx;  ///< 2 site Pauli-y operator in z x basis
extern Eigen::Matrix4cd z_xy;  ///< 2 site Pauli-z operator in x y basis
extern Eigen::Matrix4cd z_yx;  ///< 2 site Pauli-z operator in y x basis
/**
 * @brief The Honeycomb Kitaev Model.
 */
class kitaevS3 {
    lattice::honeycombS3 lat;        ///< Honeycomb lattice object.
    operators::bond_op hamiltonian;  ///< Bond Operator Hamiltonian object.
   public:
    /**
     * @brief Kitaev model consturctor.
     *
     * @param size Number of unitcells in one direction.
     * @param J the coupling in all three directions
     */
    kitaevS3(size_t size, double J) : kitaevS3(size, {J, J, J}) {}
    /**
     * @brief Kitaev model constructor.
     *
     * @param size Number of unitcells in one direction.
     * @param J Array of coupling constant {J_x, J_y, J_z}..
     */
    kitaevS3(size_t size, const std::array<double, 3>& J);

    /**
     * @brief Hamiltonian constructor
     *
     * @return Reference to the `bond_op` Hamiltonian.
     */
    operators::base_op& get_hamiltonian() { return hamiltonian; }
    /**
     * @brief Lattice getter
     *
     * @return Reference to the `honeycomb` lattice.
     */
    lattice::bravais& get_lattice() { return lat; };
};
}  // namespace model
