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

#include <Eigen/Sparse>
#include <complex>
#include <memory>
//
#include <lattice/bravais.hpp>
#include <operators/bond_op.hpp>
#include <operators/local_op_chain.hpp>

namespace model {

typedef Eigen::SparseMatrix<std::complex<double>> SparseXcd;

extern SparseXcd kron(const std::vector<SparseXcd>&);
extern SparseXcd sx();
extern SparseXcd sy();
extern SparseXcd sz();

class abstract_model {
   protected:
    std::unique_ptr<lattice::bravais> lattice_;  ///< lattice object pointer.
    std::unique_ptr<operators::local_op_chain>
        hamiltonian_;  ///< Bond Operator Pointer.

    size_t helpers_ = 0;

   public:
    virtual ~abstract_model() = default;

    /**
     * @brief Hamiltonian constructor
     *
     * @return Reference to the `bond_op` Hamiltonian.
     */
    operators::local_op_chain& get_hamiltonian() { return *hamiltonian_; }
    /**
     * @brief Lattice getter
     *
     * @return Reference to the `honeycomb` lattice.
     */
    lattice::bravais& get_lattice() { return *lattice_; };

    virtual bool supports_helper_hamiltonian() const { return false; }

    virtual void add_helper_hamiltonian(double strength) {}

    virtual void remove_helper_hamiltoian();
};
}  // namespace model
