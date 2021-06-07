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
#include <chrono>
#include <cmath>
#include <complex>
#include <tools/logger.hpp>
#include <unsupported/Eigen/IterativeSolvers>
#include <vector>
//
#include <machine/abstract_sampler.hpp>
#include <machine/rbm_base.hpp>
#include <operators/base_op.hpp>
#include <operators/derivative_op.hpp>
#include <optimizer/outer_matrix.hpp>
#include <optimizer/plugin.hpp>
#include <optimizer/stochastic_reconfiguration.hpp>
#include <tools/ini.hpp>

using namespace optimizer;

stochastic_reconfiguration::stochastic_reconfiguration(
    machine::rbm_base& rbm, machine::abstract_sampler& sampler,
    operators::base_op& hamiltonian, const ini::decay_t& lr,
    const ini::decay_t& kp, bool use_gmres)
    : Base{rbm, sampler, hamiltonian, lr},
      // Initialize SR aggregators
      use_gmres_{use_gmres},
      a_dh_{derivative_, hamiltonian_},
      a_dd_{use_gmres_ ? (std::unique_ptr<operators::aggregator>)
                             std::make_unique<operators::outer_aggregator_lazy>(
                                 derivative_, sampler.get_n_samples())
                       : (std::unique_ptr<operators::aggregator>)
                             std::make_unique<operators::outer_aggregator>(
                                 derivative_)},
      // Initialize the regularization.
      kp_{kp, rbm_.get_n_updates()} {}

void stochastic_reconfiguration::register_observables() {
    // Register operators and aggregators
    Base::register_observables();
    sampler_.register_aggs({&a_dh_, a_dd_.get()});
}

void stochastic_reconfiguration::optimize() {
    // Get the result
    auto& h = a_h_.get_result();
    auto& d = a_d_.get_result();
    auto& dh = a_dh_.get_result();

    // Log energy, energy variance and sampler properties.
    logger::log(std::real(h(0)) / rbm_.n_visible, "Energy");
    logger::log(std::real(a_h_.get_variance()(0)) / rbm_.n_visible,
                "EnergyVariance");
    // logger::log(std::abs(std::imag(h(0))), "EnergyImag");
    sampler_.log();

    Eigen::MatrixXcd dw(rbm_.get_n_params(), 1);
    double reg = kp_.get();

    // Calculate Gradient
    Eigen::MatrixXcd F = dh - h(0) * d.conjugate();


    if (use_gmres_) {
        // Get covariance matrix in GMRES form
        
        OuterMatrix S =
            dynamic_cast<operators::outer_aggregator_lazy*>(a_dd_.get())
                ->construct_outer_matrix(a_d_, reg);

        Eigen::GMRES<OuterMatrix, Eigen::IdentityPreconditioner> gmres;
        gmres.compute(S);
        dw = gmres.solve(F);

    } else {
        auto& dd = a_dd_->get_result();
        // Calculate Covariance matrix.
        Eigen::MatrixXcd S = dd - d.conjugate() * d.transpose();
        // Add regularization.
        S += reg * Eigen::MatrixXcd::Identity(S.rows(), S.cols());
        dw = S.completeOrthogonalDecomposition().solve(F);
    }

    // Apply plugin if set
    if (!plug_) {
        dw *= lr_.get();
    } else {
        dw = lr_.get() * plug_->apply(dw);
    }

    // Update the weights.
    rbm_.update_weights(dw);
}
