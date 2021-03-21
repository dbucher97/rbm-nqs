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
#include <random>
//
#include <lattice/bravais.hpp>
#include <machine/rbm_base.hpp>

namespace machine {

/**
 * @brief The RBM implementation which impies the lattice translational
 * symmetry. Inherits from RMB Base and is used the same way.
 */
class rbm_symmetry : public rbm_base {
    using Base = rbm_base;

    /**
     * @brief The vector of Permutations Matrices which are eqivalent to the
     * translational symmetry of the lattice.
     */
    std::vector<Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>>
        symmetry_;

   public:
    rbm_symmetry(size_t, lattice::bravais&);

    /**
     * @brief Returns the symmtery permutations.
     *
     * @return The Reference to the Symmetry Permutations vector.
     */
    std::vector<Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>>&
    get_symmetry() {
        return symmetry_;
    };

    /**
     * @brief Returns the number of total symmtery permutations.
     *
     * @return number of symmtery permutations.
     */
    size_t symmetry_size() const { return symmetry_.size(); };

    virtual std::complex<double> psi(const Eigen::MatrixXcd& state,
                                     const Eigen::MatrixXcd&) const override;

    virtual Eigen::MatrixXcd get_thetas(
        const Eigen::MatrixXcd& state) const override;

    virtual void update_thetas(const Eigen::MatrixXcd& state,
                               const std::vector<size_t>& flips,
                               Eigen::MatrixXcd& thetas) const override;

    virtual std::complex<double> log_psi_over_psi(
        const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
        const Eigen::MatrixXcd& thetas,
        Eigen::MatrixXcd& updated_thetas) const override;

    virtual Eigen::MatrixXcd derivative(
        const Eigen::MatrixXcd& state,
        const Eigen::MatrixXcd& thetas) const override;
};

}  // namespace machine
