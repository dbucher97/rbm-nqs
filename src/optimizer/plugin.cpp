/**
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

Eigen::MatrixXcd adam_plugin::apply(Eigen::MatrixXcd& dw) {
    m_ = beta1_ * m_ + (1 - beta1_) * dw;
    wr_ = beta2_ * wr_ +
          (1 - beta2_) * (Eigen::MatrixXd)(dw.real().array().pow(2));
    wi_ = beta2_ * wi_ +
          (1 - beta2_) * (Eigen::MatrixXd)(dw.imag().array().pow(2));
    Eigen::MatrixXcd m = m_ / (1 - std::pow(beta1_, t_));
    Eigen::MatrixXcd wr = wr_ / (1 - std::pow(beta2_, t_));
    Eigen::MatrixXcd wi = wi_ / (1 - std::pow(beta2_, t_));
    t_++;
    return m.real().array() / (wr.array().sqrt() + eps_) +
           std::complex<double>(0, 1) * m.imag().array() /
               (wi.array().sqrt() + eps_);
}

momentum_plugin::momentum_plugin(size_t l, double alpha)
    : Base{}, alpha_{alpha}, m_(l, 1) {
    m_.setZero();
}

Eigen::MatrixXcd momentum_plugin::apply(Eigen::MatrixXcd& dw) {
    m_ = alpha_ * m_ + dw;
    return m_;
}
