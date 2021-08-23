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
#include <optimizer/direct_solver.hpp>
#include <optimizer/minres_solver.hpp>
#include <optimizer/outer_matrix.hpp>
#include <optimizer/plugin.hpp>
#include <optimizer/stochastic_reconfiguration.hpp>

using namespace optimizer;

stochastic_reconfiguration::stochastic_reconfiguration(
    machine::abstract_machine& rbm, sampler::abstract_sampler& sampler,
    operators::base_op& hamiltonian, const ini::decay_t& lr,
    const ini::decay_t& kp1, const ini::decay_t& kp2, const ini::decay_t& kp1d,
    std::string method, size_t max_iterations, double rtol, bool resample,
    double alpha1, double alpha2, double alpha3)
    : Base{rbm, sampler, hamiltonian, lr, resample, alpha1, alpha2, alpha3},
      // Initialize SR aggregators
      method_{method},
      max_iterations_{max_iterations},
      rtol_{rtol},
      a_dh_{derivative_, hamiltonian_},
      a_dd_{derivative_, sampler.get_my_n_samples()},
      // Initialize the regularization.
      kp1_{kp1, rbm_.get_n_updates()},
      kp2_{kp2, rbm_.get_n_updates()},
      kp1d_{kp1d, rbm_.get_n_updates()},
      F_(rbm.get_n_params()),
      dw_(rbm.get_n_params(), 1) {
    if (method == "direct") {
        solver_ = std::make_unique<direct_solver>(rbm_.get_n_params(),
                                                  rbm_.get_n_neural_params());
    } else if (method == "minres") {
        solver_ = std::make_unique<minres_solver>(
            rbm_.get_n_params(), sampler.get_n_samples(),
            sampler.get_my_n_samples(), rbm_.get_n_neural_params(),
            max_iterations, rtol);
    }
}

void stochastic_reconfiguration::register_observables() {
    // Register operators and aggregators
    Base::register_observables();
    sampler_.register_aggs({&a_dh_, &a_dd_});
}

Eigen::MatrixXcd stochastic_reconfiguration::gradient(bool log) {
    // Get the result
    auto& h = a_h_.get_result();
    auto& d = a_d_.get_result();
    auto& dh = a_dh_.get_result();
    a_dd_.finalize_diag(d);

    if (log) {
        // Log energy, energy variance and sampler properties.
        logger::log(std::real(h(0)) / rbm_.n_visible, "Energy");
        logger::log(std::real(a_h_.get_variance()(0)) / rbm_.n_visible,
                    "Energy Variance");
        // logger::log(std::abs(std::imag(h(0))), "EnergyImag");
        sampler_.log();
    }

    double reg1 = kp1_.get();
    double reg2 = kp2_.get();
    double reg1delta = kp1d_.get();
    std::cout << reg1 << ", " << reg2 << ", " << reg1delta << std::endl;

    // Calculate Gradient
    F_ = dh - h(0) * d.conjugate();

    dw_.setZero();
    solver_->solve(a_dd_.get_result(), d, a_dd_.get_norm(), F_, dw_, reg1, reg2,
                   reg1delta, a_dd_.get_diag());

    if (plug_) {
        plug_->add_metric(&(a_dd_.get_result()), &a_d_.get_result());
    }
    // if (method_ == "minresqlp") {
    // Get covariance matrix in GMRES form

    /* OuterMatrix S =
    dynamic_cast<operators::outer_aggregator_lazy*>(a_dd_.get())
            ->construct_outer_matrix(a_d_, reg1, reg2);

    Eigen::ConjugateGradient<OuterMatrix, Eigen::Upper | Eigen::Lower,
                             Eigen::IdentityPreconditioner>
        cg;
    if (max_iterations_) cg.setMaxIterations(max_iterations_);
    cg.compute(S);
    dw = cg.solve(F); */

    // auto oa =
    // dynamic_cast<operators::outer_aggregator_lazy*>(a_dd_.get());
    // minresqlp_adapter min(oa->get_result(), a_d_.get_result(), reg1,
    // reg2,
    //                       reg1delta, oa->get_norm(),
    //                       rbm_.get_n_neural_params());
    // if (max_iterations_) min.itnlim = max_iterations_;
    // if (rtol_ > 0.0) min.rtol = rtol_;
    // min.apply(F, dw);
    /* std::cout << std::endl;
    std::cout << "Acond: " << min.getAcond() << std::endl;
    std::cout << "Rnorm: " << min.getRnorm() << std::endl;
    std::cout << "Rtol: " << min.rtol << std::endl;
    std::cout << "Itn: " << min.getItn() << std::endl;
    std::cout << "Istop: " << min.getIstop() << std::endl; */
    // } else if (method_ == "direct") {
    //     auto& dd = a_dd_->get_result();
    //     // Calculate Covariance matrix.
    //     Eigen::MatrixXcd S = dd - d.conjugate() * d.transpose();
    //     // Add regularization.
    //     auto Sdiag = S.diagonal();
    //     S += Eigen::DiagonalMatrix<std::complex<double>,
    //     Eigen::Dynamic>(reg1 *
    //                                                                      Sdiag);
    //     S += reg2 * Sdiag.cwiseAbs().maxCoeff() *
    //          Eigen::MatrixXcd::Identity(S.rows(), S.cols());
    //     dw = S.completeOrthogonalDecomposition().solve(F);
    // } else {
    //     throw std::runtime_error("SR method '" + method_ + "' does not
    //     exist!");
    // }

    // logger::log(dw.norm() / dw.size(), "DW Norm");
    return dw_;
}
