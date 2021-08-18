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

class file_psi : public abstract_machine {
    using Base = abstract_machine;

    Eigen::MatrixXcd state_vec_;

   public:
    file_psi(lattice::bravais& lattice, const std::string& filename);

    virtual rbm_context get_context(
        const Eigen::MatrixXcd& state) const override {
        return {Eigen::MatrixXcd::Zero(1, 1), 0};
    }

    virtual Eigen::MatrixXcd derivative(const Eigen::MatrixXcd&,
                                        const rbm_context&) const override {
        return Eigen::MatrixXcd::Zero(1, 1);
    }

    virtual void update_context(const Eigen::MatrixXcd& state,
                                const std::vector<size_t>& flips,
                                rbm_context& thetas) const override {}

    virtual std::complex<double> psi(const Eigen::MatrixXcd& state,
                                     rbm_context& context) const override;
    virtual std::complex<double> psi_over_psi(
        const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
        rbm_context& context, rbm_context& updated_context) const override;
};

}  // namespace machine
