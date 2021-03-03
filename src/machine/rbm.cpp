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
#include <machine/rbm.hpp>
#include <math.hpp>

using namespace machine;

rbm::rbm(size_t n_alpha, lattice::bravais& l)
    : n_alpha{n_alpha},
      n_visible{l.n_total},
      lattice_{l},
      weights_(n_visible, n_alpha),
      h_bias_(n_alpha, 1),
      v_bias_{0},
      symmetry_{lattice_.construct_symmetry()} {}

void rbm::initialize_weights(std::mt19937& rng, double std_dev,
                             double std_dev_imag) {
    if (std_dev_imag < 0) std_dev_imag = std_dev;
    std::normal_distribution<double> real_dist{0, std_dev};
    std::normal_distribution<double> imag_dist{0, std_dev_imag};

    v_bias_ = std::complex<double>(real_dist(rng), imag_dist(rng));
    for (size_t i = 0; i < n_alpha; i++) {
        h_bias_(i) = std::complex<double>(real_dist(rng), imag_dist(rng));
        for (size_t j = 0; j < n_visible; j++) {
            weights_(j, i) =
                std::complex<double>(real_dist(rng), imag_dist(rng));
        }
    }
}

void rbm::update_weights(const Eigen::MatrixXcd& dw) {
    v_bias_ -= dw(0);
    h_bias_ -= dw.block(1, 0, n_alpha, 1);
    Eigen::MatrixXcd dww = dw.block(1 + n_alpha, 0, n_alpha * n_visible, 1);
    weights_ -= Eigen::Map<Eigen::MatrixXcd>(dww.data(), n_visible, n_alpha);
    // std::cout << v_bias_ << std::endl;
    // std::cout << h_bias_ << std::endl;
    // std::cout << weights_ << std::endl;
}

std::complex<double> rbm::psi(const Eigen::MatrixXcd& state,
                              const Eigen::MatrixXcd& thetas) {
    std::complex<double> cosh_part = lncosh(thetas).sum();
    return std::exp(cosh_part) * (v_bias_ * state).array().exp().prod();
}

Eigen::MatrixXcd rbm::get_thetas(const Eigen::MatrixXcd& state) const {
    Eigen::MatrixXcd ret(n_alpha, symmetry_.size());
    for (size_t s = 0; s < symmetry_.size(); s++) {
        ret.col(s) = weights_.transpose() * (symmetry_[s] * state) + h_bias_;
    }
    return ret;
}

void rbm::update_thetas(const Eigen::MatrixXcd& state,
                        const std::vector<size_t>& flips,
                        Eigen::MatrixXcd& thetas) const {
    for (auto& f : flips) {
        for (size_t s = 0; s < symmetry_.size(); s++) {
            thetas.col(s) -=
                2 * weights_.transpose().col(symmetry_[s].indices()(f)) *
                state(f);
        }
    }
}

std::complex<double> rbm::log_psi_over_psi(
    const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
    const Eigen::MatrixXcd& thetas, Eigen::MatrixXcd& updated_thetas) const {
    if (flips.empty()) return 0.;

    std::complex<double> ret = 0;
    for (auto& f : flips) ret -= 2. * state(f);
    ret *= v_bias_;

    update_thetas(state, flips, updated_thetas);

    ret += (lncosh(updated_thetas) - lncosh(thetas)).sum();

    return ret;
}

std::complex<double> rbm::log_psi_over_psi(
    const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
    const Eigen::MatrixXcd& thetas) const {
    Eigen::MatrixXcd updated_thetas = thetas;
    return rbm::log_psi_over_psi(state, flips, thetas, updated_thetas);
}

std::complex<double> rbm::log_psi_over_psi(
    const Eigen::MatrixXcd& state, const std::vector<size_t>& flips) const {
    Eigen::MatrixXcd thetas = get_thetas(state);
    return log_psi_over_psi(state, flips, thetas);
}

std::complex<double> rbm::psi_over_psi(const Eigen::MatrixXcd& state,
                                       const std::vector<size_t>& flips,
                                       const Eigen::MatrixXcd& thetas) const {
    return std::exp(log_psi_over_psi(state, flips, thetas));
}

std::complex<double> rbm::psi_over_psi(const Eigen::MatrixXcd& state,
                                       const std::vector<size_t>& flips) const {
    Eigen::MatrixXcd thetas = get_thetas(state);
    return psi_over_psi(state, flips, thetas);
}

bool rbm::flips_accepted(double prob, const Eigen::MatrixXcd& state,
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

bool rbm::flips_accepted(double prob, const Eigen::MatrixXcd& state,
                         const std::vector<size_t>& flips) const {
    Eigen::MatrixXcd thetas = get_thetas(state);
    return flips_accepted(prob, state, flips, thetas);
}

