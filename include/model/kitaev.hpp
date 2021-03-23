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
#include <vector>
//
#include <lattice/honeycomb.hpp>
#include <operators/base_op.hpp>
#include <operators/bond_op.hpp>

/**
 * @brief Namespace for the physical models.
 */
namespace model {

extern Eigen::Matrix4cd sxsx;  ///< 2 site Pauli-x operator
extern Eigen::Matrix4cd sysy;  ///< 2 site Pauli-y operator
extern Eigen::Matrix4cd szsz;  ///< 2 site Pauli-z operator

/**
 * @brief The Honeycomb Kitaev Model.
 */
class kitaev {
    lattice::honeycomb lat;          ///< Honeycomb lattice object.
    operators::bond_op hamiltonian;  ///< Bond Operator Hamiltonian object.
   public:
    /**
     * @brief Kitaev model consturctor.
     *
     * @param size Number of unitcells in one direction.
     * @param J the coupling in all three directions
     */
    kitaev(size_t size, double J) : kitaev(size, {J, J, J}) {}
    /**
     * @brief Kitaev model constructor.
     *
     * @param size Number of unitcells in one direction.
     * @param J Array of coupling constant {J_x, J_y, J_z}..
     */
    kitaev(size_t size, const std::array<double, 3>& J)
        : lat{size},
          hamiltonian{lat.get_bonds(),
                      {J[0] * sxsx, J[1] * sysy, J[2] * szsz}} {}

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
