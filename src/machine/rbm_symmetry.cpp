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

#include <cmath>
#include <complex>
#include <iostream>
#include <random>
//
#include <machine/rbm_symmetry.hpp>
#include <math.hpp>

using namespace machine;

rbm_symmetry::rbm_symmetry(size_t n_alpha, lattice::bravais& l)
    : Base{n_alpha,
#ifndef FULL_SYMMETRY
           l.n_basis,
#else
           1,
#endif
           l},
      symmetry_{lattice_.construct_symmetry()} {
}

std::complex<double> rbm_symmetry::psi(const Eigen::MatrixXcd& state,
                                       const Eigen::MatrixXcd& thetas) const {
    // Same as base class, just modified for single `v_bias` item.
    std::complex<double> cosh_part = lncosh(thetas).sum();
#ifndef FULL_SYMMETRY
    auto vbias_part =
        v_bias_.array() * Eigen::Map<const Eigen::MatrixXcd>(
                              state.data(), n_vb_, lattice_.n_total_uc)
                              .rowwise()
                              .sum()
                              .array();
#else
    auto vbias_part = v_bias_(0) * state;
#endif
    return std::exp(cosh_part) * vbias_part.array().exp().prod();
}

std::complex<double> rbm_symmetry::psi_alt(
    const Eigen::MatrixXcd& state, const Eigen::MatrixXcd& thetas) const {
    // Same as base class, just modified for single `v_bias` item.
    std::complex<double> cosh_part = thetas.array().cosh().prod();
#ifndef FULL_SYMMETRY
    auto vbias_part =
        v_bias_.array() * Eigen::Map<const Eigen::MatrixXcd>(
                              state.data(), n_vb_, lattice_.n_total_uc)
                              .rowwise()
                              .sum()
                              .array();
#else
    auto vbias_part = v_bias_(0) * state;
#endif
    return cosh_part * vbias_part.array().exp().prod();
}

Eigen::MatrixXcd rbm_symmetry::get_thetas(const Eigen::MatrixXcd& state) const {
    // Get thetas with symmetries involved
    Eigen::MatrixXcd ret(n_alpha, symmetry_.size());
    for (size_t s = 0; s < symmetry_.size(); s++) {
        ret.col(s) = weights_.transpose() * (symmetry_[s] * state) + h_bias_;
    }
    return ret;
}

void rbm_symmetry::update_thetas(const Eigen::MatrixXcd& state,
                                 const std::vector<size_t>& flips,
                                 Eigen::MatrixXcd& thetas) const {
    // Same as base class but with symmetries involved
    Eigen::MatrixXcd state2 = state;
    for (auto& f : flips) {
        for (size_t s = 0; s < symmetry_.size(); s++) {
            thetas.col(s) -=
                2 * weights_.transpose().col(symmetry_[s].indices()(f)) *
                state2(f);
        }
        state2(f) *= -1;
    }
}

std::complex<double> rbm_symmetry::log_psi_over_psi(
    const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
    const Eigen::MatrixXcd& thetas, Eigen::MatrixXcd& updated_thetas) const {
    if (flips.empty()) return 0.;

    // Just adjusted for the single v_bias
    std::complex<double> ret = 0;
    for (auto& f : flips) ret -= 2. * state(f) * v_bias_(f % n_vb_);

    // Same as base class
    update_thetas(state, flips, updated_thetas);

    ret += (lncosh(updated_thetas) - lncosh(thetas)).sum();

    return ret;
}

std::complex<double> rbm_symmetry::psi_over_psi_alt(
    const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
    const Eigen::MatrixXcd& thetas, Eigen::MatrixXcd& updated_thetas) const {
    if (flips.empty()) return 1.;

    // Just adjusted for the single v_bias
    std::complex<double> ret = 1;
    for (auto& f : flips) ret *= std::exp(-2. * state(f) * v_bias_(f % n_vb_));

    // Same as base class
    update_thetas(state, flips, updated_thetas);

    ret *= (updated_thetas.array().cosh() / thetas.array().cosh()).prod();

    return ret;
}

Eigen::MatrixXcd rbm_symmetry::derivative(
    const Eigen::MatrixXcd& state, const Eigen::MatrixXcd& thetas) const {
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(n_params, 1);
#ifndef FULL_SYMMETRY
    result.block(0, 0, n_vb_, 1) = Eigen::Map<const Eigen::MatrixXcd>(
                                       state.data(), n_vb_, lattice_.n_total_uc)
                                       .rowwise()
                                       .sum();
#else
    result(0) = state.sum();
#endif
    // Same as baseclass
    Eigen::MatrixXcd tanh = thetas.array().tanh();
    result.block(n_vb_, 0, n_alpha, 1) = tanh.rowwise().sum();
    size_t n_tot = n_visible * n_alpha;
    // Symmetries involved
    for (size_t s = 0; s < symmetry_.size(); s++) {
        Eigen::MatrixXcd x = (symmetry_[s] * state) * tanh.col(s).transpose();
        result.block(n_vb_ + n_alpha, 0, n_tot, 1) +=
            Eigen::Map<Eigen::MatrixXcd>(x.data(), n_tot, 1);
    }
    return result;
}
