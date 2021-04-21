/*
 * Copyright (C) 2021  David Bucher <David.Bucher@physik.lmu.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <memory>
//
#include <lattice/bravais.hpp>
#include <operators/bond_op.hpp>

namespace model {

class abstract_model {
   protected:
    std::unique_ptr<lattice::bravais> lattice_;  ///< lattice object pointer.
    std::unique_ptr<operators::bond_op>
        hamiltonian_;  ///< Bond Operator Pointer.

   public:
    virtual ~abstract_model() = default;

    /**
     * @brief Hamiltonian constructor
     *
     * @return Reference to the `bond_op` Hamiltonian.
     */
    operators::bond_op& get_hamiltonian() { return *hamiltonian_; }
    /**
     * @brief Lattice getter
     *
     * @return Reference to the `honeycomb` lattice.
     */
    lattice::bravais& get_lattice() { return *lattice_; };
};
}  // namespace model
