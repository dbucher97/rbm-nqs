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

    Eigen::MatrixXcd* state_vec_;
    bool created_;

   public:
    file_psi(lattice::bravais& lattice, const std::string& filename);
    ~file_psi();
    file_psi(lattice::bravais& lattice, Eigen::MatrixXcd& state);

    virtual rbm_context get_context(const spin_state& state) const override {
        return {Eigen::MatrixXcd::Zero(1, 1)};
    }

    virtual Eigen::MatrixXcd derivative(const spin_state& state,
                                        const rbm_context&) const override {
        return Eigen::MatrixXcd::Zero(1, 1);
    }

    virtual void update_context(const spin_state& state,
                                const std::vector<size_t>& flips,
                                rbm_context& thetas) const override {}

    virtual std::complex<double> psi(const spin_state& state,
                                     rbm_context& context) override;
    virtual std::complex<double> psi_over_psi(const spin_state& state,
                                              const std::vector<size_t>& flips,
                                              rbm_context& context,
                                              rbm_context& updated_context,
                                              bool discard = false,
                                              bool* didupdate = 0) override;
};

}  // namespace machine
