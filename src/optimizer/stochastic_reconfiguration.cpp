/*
 * src/optimizer/stochastic_reconfiguration.cpp
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

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <machine/abstract_sampler.hpp>
#include <machine/rbm_base.hpp>
#include <operators/base_op.hpp>
#include <operators/derivative_op.hpp>
#include <optimizer/plugin.hpp>
#include <optimizer/stochastic_reconfiguration.hpp>
#include <tools/logger.hpp>
#include <vector>

using namespace optimizer;

stochastic_reconfiguration::stochastic_reconfiguration(
    machine::rbm_base& rbm, machine::abstract_sampler& sampler,
    operators::base_op& hamiltonian, const ini::decay_t& lr,
    const ini::decay_t& kp)
    : Base{rbm, sampler, hamiltonian, lr},
      // Initialize SR aggregators
      a_dh_{derivative_, hamiltonian_},
      a_dd_{derivative_},
      // Initialize the regularization.
      kp_{kp, rbm_.get_n_updates()} {}

void stochastic_reconfiguration::register_observables() {
    // Register operators and aggregators
    Base::register_observables();
    sampler_.register_aggs({&a_dh_, &a_dd_});
}

void stochastic_reconfiguration::optimize() {
    // Get the result
    auto& h = a_h_.get_result();
    auto& d = a_d_.get_result();
    auto& dh = a_dh_.get_result();
    auto& dd = a_dd_.get_result();

    // Log energy, energy variance and sampler properties.
    logger::log(std::real(h(0)) / rbm_.n_visible, "Energy");
    logger::log(std::real(a_h_.get_variance()(0)) / rbm_.n_visible,
                "EnergyVariance");
    // logger::log(std::abs(std::imag(h(0))), "EnergyImag");
    sampler_.log();

    // Calculate the gradient descent and the covariance matrix.
    Eigen::MatrixXcd F = dh - d.conjugate() * h(0);
    Eigen::MatrixXcd S = dd - d.conjugate() * d.transpose();

    // Add regularization.
    size_t p = rbm_.get_n_params();
    S += kp_.get() * Eigen::MatrixXcd::Identity(p, p);
    // Calculate dw.
    Eigen::MatrixXcd dw = S.completeOrthogonalDecomposition().solve(F);

    // Apply plugin if set
    if (!plug_) {
        dw *= lr_.get();
    } else {
        dw = lr_.get() * plug_->apply(dw);
    }
    // dw.real() /= 2.;

    // Update the weights.
    rbm_.update_weights(dw);
}

