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

rbm_symmetry::rbm_symmetry(size_t n_alpha, lattice::bravais& l, size_t pop_mode,
                           size_t cosh_mode)
    : Base{n_alpha, l.n_total / l.symmetry_size(), l, pop_mode, cosh_mode},
      symmetry_{lattice_.construct_symmetry()} {}

rbm_context rbm_symmetry::get_context(const Eigen::MatrixXcd& state) const {
    // Get thetas with symmetries involved
    Eigen::MatrixXcd ret(n_alpha_, symmetry_.size());
    for (size_t s = 0; s < symmetry_.size(); s++) {
        ret.col(s) = weights_.transpose() * (symmetry_[s] * state) + h_bias_;
        for (auto& c : correlators_) c->add_thetas(state, ret, s);
    }
    if (pfaffian_) {
        return {ret, pfaffian_->get_context(state)};
    } else {
        return {ret};
    }
}

void rbm_symmetry::update_context(const Eigen::MatrixXcd& state,
                                  const std::vector<size_t>& flips,
                                  rbm_context& context) const {
    Eigen::MatrixXcd& thetas = context.thetas;
    // Same as base class but with symmetries involved
    std::vector<std::vector<size_t>> cidxs;
    for (auto& c : correlators_) {
        c->get_cidxs_from_flips(flips, cidxs);
    }

    // for (const auto& f : flips) {
    // Eigen::PermutationMatrix<Eigen::Dynamic> p_mat(symmetry_.size());
    // for (size_t s = 0; s < symmetry_.size(); s++) {
    //     p_mat.indices()(s) = symmetry_[s].indices()(f);
    // }
    // thetas.noalias() -= 2. * state(f) * (weights_.transpose() * p_mat);
    // }
    for (size_t s = 0; s < symmetry_.size(); s++) {
        // Possibility for speedup thetas - perm matrix weights.transpose()
        // state
        for (auto& f : flips) {
            thetas.col(s) -=
                2 * weights_.row(symmetry_[s].indices()(f)).transpose() *
                state(f);
        }
        for (size_t i = 0; i < correlators_.size(); i++) {
            correlators_[i]->update_thetas(state, cidxs[i], thetas, s);
        }
    }

    if (pfaffian_) {
        pfaffian_->update_context(state, flips, context.pfaff());
    }
}

Eigen::MatrixXcd rbm_symmetry::derivative(const Eigen::MatrixXcd& state,
                                          const rbm_context& context) const {
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(get_n_params(), 1);
    result.block(0, 0, n_vb_, 1) = Eigen::Map<const Eigen::MatrixXcd>(
                                       state.data(), n_vb_, lattice_.n_total_uc)
                                       .rowwise()
                                       .sum();
    // Same as baseclass
    // Eigen::MatrixXcd tanh = thetas.array().tanh();
    Eigen::MatrixXcd tanh = (*tanh_)(context.thetas);
    result.block(n_vb_, 0, n_alpha_, 1) = tanh.rowwise().sum();
    size_t n_tot = n_visible * n_alpha_;
    // Symmetries involved
    for (size_t s = 0; s < symmetry_.size(); s++) {
        Eigen::MatrixXcd x = (symmetry_[s] * state) * tanh.col(s).transpose();
        result.block(n_vb_ + n_alpha_, 0, n_tot, 1) +=
            Eigen::Map<Eigen::MatrixXcd>(x.data(), n_tot, 1);
    }

    size_t offset = n_params_;
    for (auto& c : correlators_) c->derivative(state, tanh, result, offset);
    return result;
}

void rbm_symmetry::add_correlator(
    const std::vector<std::vector<size_t>>& corr) {
    auto symm = lattice_.construct_uc_symmetry();
    correlators_.push_back(std::make_unique<correlator>(corr, n_alpha_, symm));
}

std::complex<double> rbm_symmetry::psi_notheta(
    const Eigen::MatrixXcd& state) const {
    auto vbias_part =
        v_bias_.array() * Eigen::Map<const Eigen::MatrixXcd>(
                              state.data(), n_vb_, lattice_.n_total_uc)
                              .rowwise()
                              .sum()
                              .array();
    return vbias_part.array().exp().prod();
}

std::complex<double> rbm_symmetry::log_psi_over_psi(
    const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
    const rbm_context& context, rbm_context& updated_context) const {
    if (flips.empty()) return 0.;

    // Just adjusted for the single v_bias
    std::complex<double> ret = 0;
    for (auto& f : flips) ret -= 2. * state(f) * v_bias_(f % n_vb_);

    std::vector<std::vector<size_t>> cidxs;
    for (auto& c : correlators_) {
        c->get_cidxs_from_flips(flips, cidxs);
        ret += c->log_psi_over_psi(state, *(cidxs.end() - 1));
    }

    // Same as base class
    update_context(state, flips, updated_context);

    ret += (math::lncosh(updated_context.thetas) - math::lncosh(context.thetas))
               .sum();

    return ret;
}

std::complex<double> rbm_symmetry::psi_over_psi_alt(
    const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
    const rbm_context& context, rbm_context& updated_context) const {
    if (flips.empty()) return 1.;

    // Just adjusted for the single v_bias
    std::complex<double> ret = 1;
    for (auto& f : flips) ret *= std::exp(-2. * state(f) * v_bias_(f % n_vb_));

    std::vector<std::vector<size_t>> cidxs;
    for (auto& c : correlators_) {
        c->get_cidxs_from_flips(flips, cidxs);
        ret *= std::exp(c->log_psi_over_psi(state, *(cidxs.end() - 1)));
    }

    // Same as base class
    update_context(state, flips, updated_context);

    // ret *= (updated_context.thetas.array().cosh() /
    // context.thetas.array().cosh()).prod();
    ret *= ((*cosh_)(updated_context.thetas).array() /
            (*cosh_)(context.thetas).array())
               .prod();

    return ret;
}
