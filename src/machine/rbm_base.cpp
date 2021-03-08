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

#include <cmath>
#include <complex>
#include <iostream>
#include <random>
//
#include <machine/rbm_base.hpp>

using namespace machine;

rbm_base::rbm_base(size_t n_alpha, size_t n_v_bias, lattice::bravais& l)
    : n_alpha{n_alpha},
      n_visible{l.n_total},
      n_params{n_v_bias + n_alpha + n_alpha * n_visible},
      lattice_{l},
      weights_(n_visible, n_alpha),
      h_bias_(n_alpha, 1),
      v_bias_(n_v_bias, 1),
      n_vb_{n_v_bias} {}

rbm_base::rbm_base(size_t n_alpha, lattice::bravais& l)
    : rbm_base{n_alpha, l.n_total, l} {}

void rbm_base::initialize_weights(std::mt19937& rng, double std_dev,
                                  double std_dev_imag) {
    if (std_dev_imag < 0) std_dev_imag = std_dev;
    std::normal_distribution<double> real_dist{0, std_dev};
    std::normal_distribution<double> imag_dist{0, std_dev_imag};

    for (size_t i = 0; i < n_vb_; i++) {
        v_bias_(i) = std::complex<double>(real_dist(rng), imag_dist(rng));
    }
    for (size_t i = 0; i < n_alpha; i++) {
        h_bias_(i) = std::complex<double>(real_dist(rng), imag_dist(rng));
        for (size_t j = 0; j < n_visible; j++) {
            weights_(j, i) =
                std::complex<double>(real_dist(rng), imag_dist(rng));
        }
    }
}

void rbm_base::update_weights(const Eigen::MatrixXcd& dw) {
    v_bias_ -= dw.block(0, 0, n_vb_, 1);
    h_bias_ -= dw.block(n_vb_, 0, n_alpha, 1);
    Eigen::MatrixXcd dww = dw.block(n_vb_ + n_alpha, 0, n_alpha * n_visible, 1);
    weights_ -= Eigen::Map<Eigen::MatrixXcd>(dww.data(), n_visible, n_alpha);
}

std::complex<double> rbm_base::log_psi_over_psi(
    const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
    const Eigen::MatrixXcd& thetas) const {
    Eigen::MatrixXcd updated_thetas = thetas;
    return log_psi_over_psi(state, flips, thetas, updated_thetas);
}

std::complex<double> rbm_base::log_psi_over_psi(
    const Eigen::MatrixXcd& state, const std::vector<size_t>& flips) const {
    Eigen::MatrixXcd thetas = get_thetas(state);
    return log_psi_over_psi(state, flips, thetas);
}

std::complex<double> rbm_base::psi_over_psi(
    const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
    const Eigen::MatrixXcd& thetas) const {
    return std::exp(log_psi_over_psi(state, flips, thetas));
}

std::complex<double> rbm_base::psi_over_psi(
    const Eigen::MatrixXcd& state, const std::vector<size_t>& flips) const {
    Eigen::MatrixXcd thetas = get_thetas(state);
    return psi_over_psi(state, flips, thetas);
}

bool rbm_base::flips_accepted(double prob, const Eigen::MatrixXcd& state,
                              const std::vector<size_t>& flips,
                              Eigen::MatrixXcd& thetas) const {
    Eigen::MatrixXcd new_thetas = thetas;
    double a = std::exp(
        2 * std::real(log_psi_over_psi(state, flips, thetas, new_thetas)));
    if (prob < a) {
        thetas = new_thetas;
        return true;
    } else {
        return false;
    }
}

bool rbm_base::flips_accepted(double prob, const Eigen::MatrixXcd& state,
                              const std::vector<size_t>& flips) const {
    Eigen::MatrixXcd thetas = get_thetas(state);
    return flips_accepted(prob, state, flips, thetas);
}

