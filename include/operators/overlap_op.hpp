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

#include <Eigen/Dense>
#include <complex>
#include <string>
#include <vector>
//
#include <operators/base_op.hpp>

namespace operators {

/**
 * @brief Calculates the overlap of a RBM state with a quantum state from a
 * file.
 */
class overlap_op : public base_op {
    using Base = base_op;
    Eigen::MatrixXcd
        state_vec_;  ///< The state vector of the loaded quantum state.
    size_t n_vis_;   ///< Number of spins.

    /**
     * @brief Returns the \psi(\sigma) of the loaded quantum state
     *
     * @param state The current state.
     *
     * @return \psi(\sigma) of the loaded quantum state.
     */
    std::complex<double> get_psi(const Eigen::MatrixXcd& state);

    /**
     * @brief Fills the `state_vec_` from a file.
     *
     * @param file Name of the file.
     */
    void fill_vec(const std::string& file);

   public:
    /**
     * @brief Overlap operator constructor.
     *
     * @param file Filename of the quantum state (stored as a list of complex
     * numebers).
     * @param n_visible Number of sites.
     */
    overlap_op(const std::string& file, size_t n_visible);

    virtual void evaluate(machine::abstract_machine&, const Eigen::MatrixXcd&,
                          machine::rbm_context&) override;
};

}  // namespace operators
