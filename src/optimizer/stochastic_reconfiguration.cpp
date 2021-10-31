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
#include <vector>
//
#include <optimizer/cg_solver.hpp>
#include <optimizer/direct_solver.hpp>
#include <optimizer/minresqlp_solver.hpp>
#include <optimizer/outer_matrix.hpp>
#include <optimizer/plugin.hpp>
#include <optimizer/stochastic_reconfiguration.hpp>
#include <sampler/full_sampler.hpp>

using namespace optimizer;

stochastic_reconfiguration::stochastic_reconfiguration(
    machine::abstract_machine& rbm, sampler::abstract_sampler& sampler,
    operators::local_op_chain& hamiltonian, const ini::decay_t& lr,
    const ini::decay_t& kp1, const ini::decay_t& kp2, const ini::decay_t& kp1d,
    std::string method, size_t max_iterations, double rtol, bool resample,
    double alpha1, double alpha2, double alpha3)
    : Base{rbm, sampler, hamiltonian, lr, resample, alpha1, alpha2, alpha3},
      // Initialize SR aggregators
      method_{method},
      max_iterations_{max_iterations},
      rtol_{rtol},
      a_dh_{derivative_, hamiltonian_, sampler.get_my_n_samples()},
      a_dd_{derivative_, sampler.get_my_n_samples()},
      // Initialize the regularization.
      kp1_{kp1, rbm_.get_n_updates()},
      kp2_{kp2, rbm_.get_n_updates()},
      kp1d_{kp1d, rbm_.get_n_updates()},
      F_(rbm.get_n_params()) {
    if (method == "direct") {
        solver_ = std::make_unique<direct_solver>(rbm_.get_n_params(),
                                                  rbm_.get_n_neural_params());
    } else if (method == "minresqlp") {
        solver_ = std::make_unique<minresqlp_solver>(
            rbm_.get_n_params(), sampler.get_n_samples(),
            sampler.get_my_n_samples(), rbm_.get_n_neural_params(),
            max_iterations, rtol);
    } else if (method == "cg" || method == "minres" || method == "cg-single" ||
               method == "cg-dynamic") {
        solver_ = std::make_unique<cg_solver>(
            rbm_.get_n_params(), sampler.get_n_samples(),
            rbm_.get_n_neural_params(), max_iterations, rtol, method);
    }
}

void stochastic_reconfiguration::register_observables() {
    // Register operators and aggregators
    Base::register_observables();
    sampler_.register_aggs({&a_dh_, &a_dd_});
}

Eigen::VectorXcd& stochastic_reconfiguration::gradient(bool log) {
    // Get the result
    auto& h = a_h_.get_result();
    auto& d = a_d_.get_result();
    auto& dh = a_dh_.get_result();
    a_dd_.finalize_diag(d);
    double x = a_dd_.get_diag().minCoeff();
    if (x < 0) mpi::cout << x << mpi::endl;

    if (log) {
        // Log energy, energy variance and sampler properties.
        logger::log(std::real(h(0)) / rbm_.n_visible, "Energy");
        logger::log(a_h_.get_stddev()(0) / rbm_.n_visible, "Energy Stddev");
        // std::cout << "\n" << a_h_.get_tau()(0) << ", ";

        // sampler::full_sampler smp{rbm_, 2};
        // smp.register_op(&hamiltonian_);
        // operators::aggregator ah(hamiltonian_, smp.get_my_n_samples());
        // ah.track_variance();
        // smp.register_agg(&ah);
        // smp.sample();
        // auto x = ah.get_result();
        // logger::log(std::real(x(0)) / rbm_.n_visible, "Perfect Energy");
        sampler_.log();
    }

    double reg1 = kp1_.get();
    double reg2 = kp2_.get();
    // std::cout << reg1 << ", " << reg2 << std::endl;
    double reg1delta = kp1d_.get();

    // Calculate Gradient
    F_ = dh - h(0) * d.conjugate();

    dw_.setZero();
    solver_->solve(a_dd_.get_result(), d, a_dd_.get_norm(), F_, dw_, reg1, reg2,
                   reg1delta, a_dd_.get_diag());

    // mpi::cout << a_dd_.get_diag().minCoeff() << ", "
    //           << a_dd_.get_diag().maxCoeff() << mpi::endl;

    if (plug_) {
        plug_->add_metric(&(a_dd_.get_result()), &a_d_.get_result());
    }
    // logger::log(dw.norm() / dw.size(), "DW Norm");
    return dw_;
}
