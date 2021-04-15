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
//
#include <optimizer/gradient_descent.hpp>
#include <tools/logger.hpp>

using namespace optimizer;

gradient_descent::gradient_descent(machine::rbm_base& rbm,
                                   machine::abstract_sampler& sampler,
                                   operators::base_op& hamiltonian,
                                   const ini::decay_t& lr)
    : Base{rbm, sampler, hamiltonian, lr},
      // Initialize SR aggregators
      a_dh_{derivative_, hamiltonian_} {}

void gradient_descent::register_observables() {
    // Register operators and aggregators
    Base::register_observables();
    sampler_.register_agg(&a_dh_);
}

void gradient_descent::optimize() {
    // Get the result
    auto& h = a_h_.get_result();
    auto& d = a_d_.get_result();
    auto& dh = a_dh_.get_result();

    // Log energy, energy variance and sampler properties.
    logger::log(std::real(h(0)) / rbm_.n_visible, "Energy");
    logger::log(std::real(a_h_.get_variance()(0)) / rbm_.n_visible,
                "EnergyVariance");
    sampler_.log();

    // Calculate the gradient descent
    Eigen::MatrixXcd dw = dh - d.conjugate() * h(0);

    // Apply plugin if set
    if (!plug_) {
        dw *= lr_.get();
    } else {
        dw = lr_.get() * plug_->apply(dw);
    }
    dw.real() /= 2.;

    // Update the weights.
    rbm_.update_weights(dw);
}