/*
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
#include <iostream>
//
#include <optimizer/plugin.hpp>
#include <tools/mpi.hpp>

using namespace optimizer;

adam_plugin::adam_plugin(size_t l, double beta1, double beta2, double eps)
    : Base{},
      beta1_{beta1},
      beta2_{beta2},
      eps_{eps},
      t_{1},
      m_(l, 1),
      wr_(l, 1),
      wi_(l, 1) {
    m_.setZero();
    wr_.setZero();
    wi_.setZero();
}

void adam_plugin::apply(Eigen::VectorXcd& dw, double lr) {
    // Calcualte the Adam optimization,
    // see https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam
    m_ = beta1_ * m_ + (1 - beta1_) * dw;
    // Split second moment into real and imaginary part.
    wr_ = beta2_ * wr_ +
          (1 - beta2_) * (Eigen::MatrixXd)(dw.real().array().pow(2));
    wi_ = beta2_ * wi_ +
          (1 - beta2_) * (Eigen::MatrixXd)(dw.imag().array().pow(2));
    Eigen::MatrixXcd m = m_ / (1 - std::pow(beta1_, t_));
    Eigen::MatrixXcd wr = wr_ / (1 - std::pow(beta2_, t_));
    Eigen::MatrixXcd wi = wi_ / (1 - std::pow(beta2_, t_));
    t_++;
    dw = m.real().array() / (wr.array().sqrt() + eps_) +
         std::complex<double>(0, 1) * m.imag().array() /
             (wi.array().sqrt() + eps_);
    dw *= lr;
}

momentum_plugin::momentum_plugin(size_t l, double alpha, double dialup)
    : Base{},
      alpha_{alpha},
      dialup_{dialup},
      dup_{dialup_ == 1. ? 1. : 1e-3},
      m_(l, 1) {
    m_.setZero();
}

void momentum_plugin::apply(Eigen::VectorXcd& dw, double lr) {
    // Do the momentum update step.
    double ax = alpha_;
    if (m_.isZero()) m_ = dw;
    if (dup_ < 1.) {
        dup_ *= dialup_;
        ax *= dup_;
    }

    m_ = ax * m_ + (1 - ax) * dw;
    dw = lr * m_;
}

heun_plugin::heun_plugin(const std::function<Eigen::VectorXcd&(void)>& gradient,
                         machine::abstract_machine& rbm,
                         sampler::abstract_sampler& sampler, double eps)
    : Base{}, gradient_{gradient}, rbm_{rbm}, sampler_{sampler}, eps_{eps} {}

void heun_plugin::apply(Eigen::VectorXcd& dw, double lr) {
    Eigen::MatrixXcd delta = 0.5 * lr * dw;
    rbm_.update_weights_nc(delta);
    sampler_.sample();
    delta -= 0.5 * lr * gradient_();
    delta /= 6;
    double x;

    if (met1_ && met2_) {
        std::complex<double> x2 = ((*met2_).transpose() * delta)(0);
        x = ((delta.transpose() * *met1_).transpose().array() - x2)
                .abs2()
                .sum();
        x = std::sqrt(x) / delta.size();
    } else {
        x = delta.norm();
    }
    //    std::cout << std::endl;
    //    std::cout << x << ", " << delta.norm() << ", " << delta.norm() / x
    //              << std::endl;

    x = std::pow(eps_ / x, 1. / 3);
    //    std::cout << x << std::endl;
    dw *= (x - 0.5) * lr;
}
