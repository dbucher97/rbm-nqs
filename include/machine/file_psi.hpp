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
#include <machine/rbm_base.hpp>

namespace machine {

class file_psi : public rbm_base {
    using Base = rbm_base;

    Eigen::MatrixXcd state_vec_;

   public:
    file_psi(lattice::bravais& lattice, const std::string& filename);

    virtual std::complex<double> psi(const Eigen::MatrixXcd& state,
                                     const Eigen::MatrixXcd&) const override;

    virtual std::complex<double> psi_over_psi(
        const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
        const Eigen::MatrixXcd&) const override;

    virtual std::complex<double> psi_over_psi_alt(
        const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
        const Eigen::MatrixXcd& thetas, Eigen::MatrixXcd&) const override {
        return psi_over_psi(state, flips, thetas);
    }

    virtual std::complex<double> psi_alt(
        const Eigen::MatrixXcd& state,
        const Eigen::MatrixXcd& t) const override {
        return psi(state, t);
    }

    virtual Eigen::MatrixXcd get_thetas(
        const Eigen::MatrixXcd& state) const override {
        return Eigen::MatrixXcd::Zero(1, 1);
    }

    virtual void update_thetas(const Eigen::MatrixXcd& state,
                               const std::vector<size_t>& flips,
                               Eigen::MatrixXcd& thetas) const override {}
};

}  // namespace machine
