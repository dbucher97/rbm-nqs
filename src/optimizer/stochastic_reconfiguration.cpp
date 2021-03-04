/**
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
#include <vector>
//
#include <machine/abstract_sampler.hpp>
#include <machine/rbm.hpp>
#include <operators/base_op.hpp>
#include <operators/derivative_op.hpp>
#include <optimizer/stochastic_reconfiguration.hpp>

using namespace optimizer;

stochastic_reconfiguration::stochastic_reconfiguration(
    machine::rbm& rbm, machine::abstract_sampler& sampler,
    operators::base_op& hamiltonian, double lr, double k0, double kmin,
    double m)
    : rbm_{rbm},
      sampler_{sampler},
      hamiltonian_{hamiltonian},
      derivative_{rbm.n_alpha, rbm.get_symmetry().size()},
      a_h_{hamiltonian_},
      a_d_{derivative_},
      a_dh_{derivative_, hamiltonian_},
      a_dd_{derivative_},
      n_total_{1 + rbm.n_alpha * (1 + rbm.get_symmetry().size())},
      lr_{lr},
      m_{m},
      kmin_{kmin},
      kp_{k0} {}

void stochastic_reconfiguration::register_observables() {
    sampler_.register_ops({&hamiltonian_, &derivative_});
    sampler_.register_aggs({&a_h_, &a_d_, &a_dh_, &a_dd_});
}

void stochastic_reconfiguration::optimize() {
    auto& h = a_h_.get_result();
    std::cout << h / rbm_.n_visible << std::endl;
    auto& d = a_d_.get_result();
    auto& dh = a_dh_.get_result();
    auto& dd = a_dd_.get_result();
    Eigen::MatrixXcd F = dh - d.conjugate() * h(0);
    Eigen::MatrixXcd S = dd - d.conjugate() * d.transpose();

    S += kp_ * Eigen::MatrixXcd::Identity(n_total_, n_total_);
    if (!reg_min_) {
        kp_ *= m_;
        if (kp_ < kmin_) {
            kp_ = kmin_;
            reg_min_ = true;
        }
    }
    Eigen::MatrixXcd dw =
        lr_ * S.completeOrthogonalDecomposition().pseudoInverse() * F;
    // Eigen::MatrixXcd dw = lr_ * F;

    // dws_.push_back(Eigen::MatrixXcd(dw));
    //
    // Eigen::MatrixXcd dwp(dw);
    // dwp.setZero();
    // double f = 1.;
    // for (size_t i = dws_.size() - 1; i < dws_.size(); i--) {
    //     f *= 0.5;
    //     if (f > 1e-6) {
    //         dwp += f * dws_[i];
    //     } else
    //         break;
    // }

    rbm_.update_weights(dw);
}
