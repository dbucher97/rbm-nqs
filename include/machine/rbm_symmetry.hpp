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
    std::vector<Eigen::PermutationMatrix<Eigen::Dynamic>> symmetry_;

    std::vector<Eigen::PermutationMatrix<Eigen::Dynamic>> reverse_symm_;

   public:
    rbm_symmetry(size_t, lattice::bravais&, size_t pop_mode = 0,
                 size_t cosh_mode = 0);

    virtual size_t symmetry_size() const override { return symmetry_.size(); };

    virtual rbm_context get_context(const spin_state& state) const override;

    virtual void update_context(const spin_state& state,
                                const std::vector<size_t>& flips,
                                rbm_context& context) const override;

    virtual Eigen::MatrixXcd derivative(
        const spin_state& state, const rbm_context& context) const override;

    virtual void add_correlator(
        const std::vector<std::vector<size_t>>& corr) override;

   protected:
    virtual std::complex<double> psi_notheta(
        const spin_state& state) const override;

    // virtual std::complex<double> log_psi_over_psi(
    //     const spin_state& state, const std::vector<size_t>& flips,
    //     rbm_context& context, rbm_context& updated_context) override;

    // virtual std::complex<double> psi_over_psi_alt(
    //     const spin_state& state, const std::vector<size_t>& flips,
    //     rbm_context& context, rbm_context& updated_context) override;
};

}  // namespace machine
