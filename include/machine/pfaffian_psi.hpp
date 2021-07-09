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

#include <Eigen/Dense>
#include <complex>
#include <string>
#include <vector>
//
#include <machine/abstract_machine.hpp>

namespace machine {

class pfaffian_psi : public abstract_machine {
    using Base = abstract_machine;

    Eigen::MatrixXcd state_vec_;

   public:
    pfaffian_psi(lattice::bravais& lattice) : Base{lattice, 0} {}

    virtual inline rbm_context get_context(
        const Eigen::MatrixXcd& state) const override {
        return {Eigen::MatrixXcd::Zero(1, 1), pfaffian_->get_context(state)};
    }

    virtual inline Eigen::MatrixXcd derivative(
        const Eigen::MatrixXcd& state,
        const rbm_context& context) const override {
        Eigen::MatrixXcd ret(pfaffian_->get_n_params(), 1);
        size_t offset = 0;
        pfaffian_->derivative(state, context.pfaff(), ret, offset);
        return ret;
    }

    virtual inline void update_weights(const Eigen::MatrixXcd& dw) override {
        size_t offset = 0;
        pfaffian_->update_weights(dw, offset);
    }

    virtual inline void update_context(const Eigen::MatrixXcd& state,
                                       const std::vector<size_t>& flips,
                                       rbm_context& context) const override {
        pfaffian_->update_context(state, flips, context.pfaff());
    }

    virtual inline std::complex<double> psi(
        const Eigen::MatrixXcd& state,
        const rbm_context& context) const override {
        return pfaffian_->psi(state, context.pfaff());
    }

    virtual inline std::complex<double> psi_over_psi(
        const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
        const rbm_context& context,
        rbm_context& updated_context) const override {
        return pfaffian_->psi_over_psi(state, flips, updated_context.pfaff());
    }
};

}  // namespace machine
