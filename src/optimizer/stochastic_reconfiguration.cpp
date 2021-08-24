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
#include <optimizer/minres_adapter.hpp>
#include <optimizer/outer_matrix.hpp>
#include <optimizer/plugin.hpp>
#include <optimizer/stochastic_reconfiguration.hpp>

using namespace optimizer;

stochastic_reconfiguration::stochastic_reconfiguration(
    machine::abstract_machine& rbm, sampler::abstract_sampler& sampler,
    operators::base_op& hamiltonian, const ini::decay_t& lr,
    const ini::decay_t& kp1, const ini::decay_t& kp2, const ini::decay_t& kp1d,
    bool iterative, size_t max_iterations, double rtol, bool resample,
    double alpha1, double alpha2, double alpha3)
    : Base{rbm, sampler, hamiltonian, lr, resample, alpha1, alpha2, alpha3},
      // Initialize SR aggregators
      iterative_{iterative},
      max_iterations_{max_iterations},
      rtol_{rtol},
      a_dh_{derivative_, hamiltonian_},
      a_dd_{iterative_ ? (std::unique_ptr<operators::aggregator>)
                             std::make_unique<operators::outer_aggregator_lazy>(
                                 derivative_, sampler.get_n_samples())
                       : (std::unique_ptr<operators::aggregator>)
                             std::make_unique<operators::outer_aggregator>(
                                 derivative_)},
      // Initialize the regularization.
      kp1_{kp1, rbm_.get_n_updates()},
      kp2_{kp2, rbm_.get_n_updates()},
      kp1d_{kp1d, rbm_.get_n_updates()} {}

void stochastic_reconfiguration::register_observables() {
    // Register operators and aggregators
    Base::register_observables();
    sampler_.register_aggs({&a_dh_, a_dd_.get()});
}

Eigen::MatrixXcd stochastic_reconfiguration::gradient(bool log) {
    // Get the result
    auto& h = a_h_.get_result();
    auto& d = a_d_.get_result();
    auto& dh = a_dh_.get_result();

    if (log) {
        // Log energy, energy variance and sampler properties.
        logger::log(std::real(h(0)) / rbm_.n_visible, "Energy");
        logger::log(std::real(a_h_.get_variance()(0)) / rbm_.n_visible,
                    "Energy Variance");
        // logger::log(std::abs(std::imag(h(0))), "EnergyImag");
        sampler_.log();
    }

    Eigen::MatrixXcd dw(rbm_.get_n_params(), 1);

    double reg1 = kp1_.get();
    double reg2 = kp2_.get();
    double reg1delta = kp1d_.get();

    // Calculate Gradient
    VectorXcd F = dh - h(0) * d.conjugate();

    if (iterative_) {
        auto oa = dynamic_cast<operators::outer_aggregator_lazy*>(a_dd_.get());
        const Eigen::VectorXcd diag =
            oa->get_result().cwiseAbs2().rowwise().sum() / oa->get_norm() -
            d.cwiseAbs2();
        Eigen::VectorXcd tmp(oa->get_result().cols());
        const Eigen::MatrixXcd d2 = d.conjugate().eval();

        minresqlp_adapter min(oa->get_result(), d2, reg1, reg2, reg1delta,
                              oa->get_norm(), rbm_.get_n_neural_params(), diag,
                              tmp);

        if (max_iterations_) min.itnlim = max_iterations_;
        if (rtol_ > 0.0) min.rtol = rtol_;

        min.apply(F, dw);

        // std::cout << std::endl;
        // std::cout << "Acond: " << min.getAcond() << std::endl;
        // std::cout << "Rnorm: " << min.getRnorm() << std::endl;
        // std::cout << "Rtol: " << min.rtol << std::endl;
        // std::cout << "Itn: " << min.getItn() << std::endl;
        // std::cout << "Istop: " << min.getIstop() << std::endl;
        if (plug_) {
            plug_->add_metric(&(oa->get_result()), &a_d_.get_result());
        }
    } else {
        auto& dd = a_dd_->get_result();
        // Calculate Covariance matrix.
        Eigen::MatrixXcd S = dd - d.conjugate() * d.transpose();
        // Add regularization.
        auto Sdiag = S.diagonal();
        S.diagonal().topRows(rbm_.get_n_neural_params()) *= (1 + reg1);
        S.diagonal().bottomRows(rbm_.get_n_params() -
                                rbm_.get_n_neural_params()) *=
            (1 + reg1 + reg1delta);
        S += reg2 * Sdiag.cwiseAbs().maxCoeff() *
             Eigen::MatrixXcd::Identity(S.rows(), S.cols());
        double Sd = 0, Sod = 0;
        int acc = 0;
        for (size_t i = 0; i < static_cast<size_t>(S.rows()); i++) {
            Sd = 0;
            Sod = 0;
            for (size_t j = 0; j < static_cast<size_t>(S.cols()); j++) {
                if (i == j) {
                    Sd = std::abs(S(i, i));
                } else {
                    Sod += std::abs(S(i, j));
                }
            }
            acc += Sod < Sd;
        }
        dw = S.completeOrthogonalDecomposition().solve(F);
    }

    // logger::log(dw.norm() / dw.size(), "DW Norm");
    return dw;
}
